from numpy.fft import rfft, irfft, rfftfreq
import numpy as np

rng = np.random.default_rng(42)

# Signal Generation

def harmonic_signal(n, freq=0.05, amplitude=1.0, phase=0.0):
    t = np.arange(n)
    return amplitude * np.sin(2*np.pi*freq*t + phase)

def generate_periodic_times(T, period=1.0, phase=0.0):
    """
    Deterministic periodic point process.
    Events at t = phase + k*period for k >= 0 within [0, T).
    """
    if period <= 0:
        raise ValueError("period must be positive")
    # start from the first event >= 0
    start_k = int(np.ceil((0 - phase) / period)) if phase < 0 else 0
    times = phase + np.arange(start_k, int(np.floor((T - phase) / period))) * period
    times = times[(times >= 0) & (times < T)]
    return times

def nhpp_sinusoidal_times(T, lambda0=1.0, alpha=0.8, f=1.0, rng=rng):
    """
    Thinning for sinusoidal rate:
        lambda(t) = lambda0 * (1 + alpha * sin(2π f t)), with -1 <= alpha <= 1.
    Returns event times in [0, T).
    """
    if not (-1.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [-1, 1] to keep rate nonnegative.")
    lam_max = lambda0 * (1 + abs(alpha))
    t = 0.0
    times = []
    while t < T:
        # candidate gap from Exp(lam_max)
        t += rng.exponential(1.0 / lam_max)
        if t >= T:
            break
        # acceptance probability at this time
        lam_t = lambda0 * (1.0 + alpha * np.sin(2.0 * np.pi * f * t))
        p = lam_t / lam_max  # in [ (1-|alpha|)/(1+|alpha|), 1 ]
        if rng.uniform() <= p:
            times.append(t)
    return np.array(times)

def hpp_times(T, rate=0.5, rng=rng):
    """
    Homogeneous Poisson process event times in [0, T) at constant rate.
    """
    if rate < 0:
        raise ValueError("rate must be nonnegative")
    if rate == 0:
        return np.array([])
    t = 0.0
    out = []
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= T:
            break
        out.append(t)
    return np.array(out)

def bin_events(event_times, T, dt=0.1):
    """
    Bin events into counts with bin width dt over [0, T).
    Returns (t_edges[:-1], counts).
    """
    edges = np.arange(0.0, T + 1e-12, dt)
    counts, _ = np.histogram(event_times, bins=edges)
    return edges[:-1], counts

def add_timestamp_jitter(times, jitter_std=0.01, rng=rng, kind="gaussian", half_width=None):
    """
    Add jitter to timestamps.
    kind="gaussian": N(0, jitter_std^2)
    kind="uniform": U(-half_width, +half_width)  (must set half_width)
    Clips to keep times in [0, inf) (you may later clip to [0, T)).
    """
    if kind == "gaussian":
        jitter = rng.normal(0, jitter_std, size=times.shape)
    elif kind == "uniform":
        if half_width is None:
            raise ValueError("half_width must be provided for uniform jitter")
        jitter = rng.uniform(-half_width, half_width, size=times.shape)
    else:
        raise ValueError("Unknown jitter kind: " + str(kind))
    new_times = times + jitter
    return np.clip(new_times, 0, None)

def apply_missed_detections(times, p_detect=0.9, rng=rng):
    """
    Keep each event independently with probability p_detect.
    """
    keep = rng.uniform(size=times.shape) < p_detect
    return times[keep]

def add_extraneous_events(times, T, bg_rate=0.2, rng=rng):
    """
    Superpose independent homogeneous Poisson 'clutter' with constant rate.
    Returns merged and sorted array.
    """
    extraneous_count = rng.poisson(bg_rate * T)
    if extraneous_count > 0:
        extras = rng.uniform(0, T, extraneous_count)
        all_times = np.concatenate([times, extras])
    else:
        all_times = times
    return np.sort(all_times)

def apply_noise_pipeline(times, T, jitter_std=0.02, p_detect=0.9, bg_rate=0.2, rng=rng):
    """
    Apply noise pipeline to event times:
    - Keeps events with probability p_detect,
    - Adds timestamp jitter to the detected events,
    - Superposes extraneous events (clutter).
    Returns the final noisy event times.
    """
    detected = apply_missed_detections(times, p_detect, rng)
    jittered = add_timestamp_jitter(detected, jitter_std, rng)
    final_times = add_extraneous_events(jittered, T, bg_rate, rng)
    return np.clip(final_times, 0, T)

def compute_peak_to_peak_intervals(times):
    """
    Compute differences between consecutive event times.
    """
    return np.diff(times)

def measure_hrv_error(ideal_times, noisy_times):
    """
    Measure error in heart rate variability (HRV) as the std deviation of the differences
    between consecutive intervals in ideal and noisy times.
    """
    ideal_intervals = compute_peak_to_peak_intervals(ideal_times)
    noisy_intervals = compute_peak_to_peak_intervals(noisy_times)
    min_len = min(len(ideal_intervals), len(noisy_intervals))
    if min_len == 0:
        return 0.0
    error = noisy_intervals[:min_len] - ideal_intervals[:min_len]
    return np.std(error)

def simulate_errors_for_rate(period, T, rng, jitter_std=0.02, p_detect=0.9, bg_rate=0.2):
    """
    One simulation -> return (hrv_err_full, hrv_err_missing)
    """
    ideal = generate_periodic_times(T, period, phase=0.0)
    noisy_full = apply_noise_pipeline(ideal, T, jitter_std=jitter_std, p_detect=p_detect, bg_rate=bg_rate, rng=rng)
    noisy_missing = apply_missed_detections(ideal, p_detect=p_detect, rng=rng)
    return (measure_hrv_error(ideal, noisy_full), measure_hrv_error(ideal, noisy_missing))

def batch_experiment(heart_rates=(50, 55, 60, 65, 70, 75, 80), n_runs=100, T=60, jitter_std=0.02, p_detect=0.9, bg_rate=0.2, out_csv="hrv_error_batch.csv"):
    """
    Run many simulations (different random seeds) and write mean ± std HRV errors for each heart-rate.
    """
    import csv
    periods = 60.0 / np.asarray(heart_rates, dtype=float)
    sum_full = np.zeros_like(periods, dtype=float)
    sum_sq_full = np.zeros_like(periods, dtype=float)
    sum_miss = np.zeros_like(periods, dtype=float)
    sum_sq_miss = np.zeros_like(periods, dtype=float)

    for run in range(n_runs):
        local_rng = np.random.default_rng(run)
        for i, period in enumerate(periods):
            err_full, err_miss = simulate_errors_for_rate(period, T, local_rng, jitter_std=jitter_std, p_detect=p_detect, bg_rate=bg_rate)
            sum_full[i] += err_full
            sum_sq_full[i] += err_full ** 2
            sum_miss[i] += err_miss
            sum_sq_miss[i] += err_miss ** 2

    mean_full = sum_full / n_runs
    mean_miss = sum_miss / n_runs
    std_full = np.sqrt(sum_sq_full / n_runs - mean_full ** 2)
    std_miss = np.sqrt(sum_sq_miss / n_runs - mean_miss ** 2)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heart_rate", "hrv_error_full_mean", "hrv_error_full_std", "hrv_error_missing_mean", "hrv_error_missing_std"])
        for hr, mf, sf, mm, sm in zip(heart_rates, mean_full, std_full, mean_miss, std_miss):
            w.writerow([hr, mf, sf, mm, sm])
    print(f"[batch_experiment] Results written to {out_csv}")

def main():
    # Run batch experiment for HRV error analysis with 5-minute simulations per experiment.
    batch_experiment(
        heart_rates=(50, 55, 60, 65, 70, 75, 80),
        n_runs=100,
        T=300,  # simulate for 300 seconds (5 minutes)
        jitter_std=0.02,
        p_detect=0.9,
        bg_rate=0.2,
        out_csv="hrv_error_batch.csv",
    )
    
if __name__ == "__main__":
    main()
