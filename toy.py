from numpy.fft import rfft, irfft, rfftfreq
import numpy as np
import os
from datetime import datetime

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

def measure_hrv_error(ideal_times, noisy_times, debug=False):
    """
    Measure error in heart rate variability (HRV) as the std deviation of the differences
    between consecutive intervals in ideal and noisy times.
    """
    ideal_intervals = compute_peak_to_peak_intervals(ideal_times)
    noisy_intervals = compute_peak_to_peak_intervals(noisy_times)
    if debug:
        os.makedirs("debug_hrv", exist_ok=True)
        filename = os.path.join("debug_hrv", datetime.now().strftime("%Y%m%d%H%M%S%f") + ".txt")
        with open(filename, "w") as f:
            f.write("Ideal intervals: " + str(ideal_intervals) + "\n")
            f.write("Noisy intervals: " + str(noisy_intervals) + "\n")
    min_len = min(len(ideal_intervals), len(noisy_intervals))
    if min_len == 0:
        return 0.0
    error = noisy_intervals[:min_len] - ideal_intervals[:min_len]
    return np.std(error)

def simulate_errors_for_rate(period, T, rng, jitter_std=0.02, p_detect=0.9, bg_rate=0.2, debug=False):
    """
    One simulation -> return (hrv_err_full, hrv_err_missing)
    """
    ideal = generate_periodic_times(T, period, phase=0.0)
    noisy_full = apply_noise_pipeline(ideal, T, jitter_std=jitter_std, p_detect=p_detect, bg_rate=bg_rate, rng=rng)
    noisy_missing = apply_missed_detections(ideal, p_detect=p_detect, rng=rng)
    noisy_jitter = add_timestamp_jitter(ideal, jitter_std, rng)
    noisy_extraneous = add_extraneous_events(ideal, T, bg_rate, rng)
    return (
        measure_hrv_error(ideal, noisy_full, debug),
        measure_hrv_error(ideal, noisy_missing, debug),
        measure_hrv_error(ideal, noisy_jitter, debug),
        measure_hrv_error(ideal, noisy_extraneous, debug)
    )

def batch_experiment(
    heart_rates=(50, 55, 60, 65, 70, 75, 80),
    n_runs=300,
    T=900,  # 15-minute simulation window
    jitter_std=0.02,
    p_detect=0.9,
    bg_rate=0.2,
    out_csv="hrv_error_batch.csv",
    debug=False,
):
    """
    Run many simulations and write mean ± std HRV errors for each heart-rate.
    Two versions are stored:
        * normalised error  (std ⋅ √N_intervals)  – removes 1/√N effect
        * raw error (original std difference)
    """
    import csv

    periods = 60.0 / np.asarray(heart_rates, dtype=float)

    # raw errors
    sum_full_raw = np.zeros_like(periods)
    sum_sq_full_raw = np.zeros_like(periods)
    sum_miss_raw = np.zeros_like(periods)
    sum_sq_miss_raw = np.zeros_like(periods)

    # normalised errors
    sum_full_norm = np.zeros_like(periods)
    sum_sq_full_norm = np.zeros_like(periods)
    sum_miss_norm = np.zeros_like(periods)
    sum_sq_miss_norm = np.zeros_like(periods)
    # raw errors for jitter and extraneous
    sum_jitter_raw = np.zeros_like(periods)
    sum_sq_jitter_raw = np.zeros_like(periods)
    sum_extraneous_raw = np.zeros_like(periods)
    sum_sq_extraneous_raw = np.zeros_like(periods)

    # normalised errors for jitter and extraneous
    sum_jitter_norm = np.zeros_like(periods)
    sum_sq_jitter_norm = np.zeros_like(periods)
    sum_extraneous_norm = np.zeros_like(periods)
    sum_sq_extraneous_norm = np.zeros_like(periods)

    # additional errors for jitter and extraneous events
    sum_jitter_raw = np.zeros_like(periods)
    sum_sq_jitter_raw = np.zeros_like(periods)
    sum_extraneous_raw = np.zeros_like(periods)
    sum_sq_extraneous_raw = np.zeros_like(periods)

    sum_jitter_norm = np.zeros_like(periods)
    sum_sq_jitter_norm = np.zeros_like(periods)
    sum_extraneous_norm = np.zeros_like(periods)
    sum_sq_extraneous_norm = np.zeros_like(periods)

    for run in range(n_runs):
        local_rng = np.random.default_rng(run)
        for i, period in enumerate(periods):
            err_full_raw, err_miss_raw, err_jitter_raw, err_extraneous_raw = simulate_errors_for_rate(
                period, T, local_rng, jitter_std=jitter_std,
                p_detect=p_detect, bg_rate=bg_rate, debug=True
            )
            # number of intervals for this period in window T
            n_int = max(int(T / period) - 1, 1)
            scale = np.sqrt(n_int)

            err_full_norm = err_full_raw * scale
            err_miss_norm = err_miss_raw * scale
            err_jitter_norm = err_jitter_raw * scale
            err_extraneous_norm = err_extraneous_raw * scale
            err_jitter_norm = err_jitter_raw * scale
            err_extraneous_norm = err_extraneous_raw * scale

            # accumulate
            sum_full_raw[i] += err_full_raw
            sum_sq_full_raw[i] += err_full_raw ** 2
            sum_miss_raw[i] += err_miss_raw
            sum_sq_miss_raw[i] += err_miss_raw ** 2
            sum_jitter_raw[i] += err_jitter_raw
            sum_sq_jitter_raw[i] += err_jitter_raw ** 2
            sum_extraneous_raw[i] += err_extraneous_raw
            sum_sq_extraneous_raw[i] += err_extraneous_raw ** 2

            sum_full_norm[i] += err_full_norm
            sum_sq_full_norm[i] += err_full_norm ** 2
            sum_miss_norm[i] += err_miss_norm
            sum_sq_miss_norm[i] += err_miss_norm ** 2
            sum_jitter_norm[i] += err_jitter_norm
            sum_sq_jitter_norm[i] += err_jitter_norm ** 2
            sum_extraneous_norm[i] += err_extraneous_norm
            sum_sq_extraneous_norm[i] += err_extraneous_norm ** 2

    # compute means & stds
    mean_full_norm = sum_full_norm / n_runs
    std_full_norm = np.sqrt(sum_sq_full_norm / n_runs - mean_full_norm ** 2)
    mean_miss_norm = sum_miss_norm / n_runs
    std_miss_norm = np.sqrt(sum_sq_miss_norm / n_runs - mean_miss_norm ** 2)
    mean_jitter_norm = sum_jitter_norm / n_runs
    std_jitter_norm = np.sqrt(sum_sq_jitter_norm / n_runs - mean_jitter_norm ** 2)
    mean_extraneous_norm = sum_extraneous_norm / n_runs
    std_extraneous_norm = np.sqrt(sum_sq_extraneous_norm / n_runs - mean_extraneous_norm ** 2)

    mean_full_raw = sum_full_raw / n_runs
    std_full_raw = np.sqrt(sum_sq_full_raw / n_runs - mean_full_raw ** 2)
    mean_miss_raw = sum_miss_raw / n_runs
    std_miss_raw = np.sqrt(sum_sq_miss_raw / n_runs - mean_miss_raw ** 2)
    mean_jitter_raw = sum_jitter_raw / n_runs
    std_jitter_raw = np.sqrt(sum_sq_jitter_raw / n_runs - mean_jitter_raw ** 2)
    mean_extraneous_raw = sum_extraneous_raw / n_runs
    std_extraneous_raw = np.sqrt(sum_sq_extraneous_raw / n_runs - mean_extraneous_raw ** 2)

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "heart_rate",
                "hrv_error_full_norm_mean",
                "hrv_error_full_norm_std",
                "hrv_error_full_mean",
                "hrv_error_full_std",
                "hrv_error_missing_norm_mean",
                "hrv_error_missing_norm_std",
                "hrv_error_missing_mean",
                "hrv_error_missing_std",
                "hrv_error_jitter_norm_mean",
                "hrv_error_jitter_norm_std",
                "hrv_error_extraneous_norm_mean",
                "hrv_error_extraneous_norm_std",
                "hrv_error_jitter_mean",
                "hrv_error_jitter_std",
                "hrv_error_extraneous_mean",
                "hrv_error_extraneous_std",
            ]
        )
        for vals in zip(
            heart_rates,
            mean_full_norm, std_full_norm, mean_full_raw, std_full_raw,
            mean_miss_norm, std_miss_norm, mean_miss_raw, std_miss_raw,
            mean_jitter_norm, std_jitter_norm, mean_extraneous_norm, std_extraneous_norm,
            mean_jitter_raw, std_jitter_raw, mean_extraneous_raw, std_extraneous_raw,
        ):
            w.writerow(vals)

    print(f"[batch_experiment] Results written to {out_csv}")

def main():
    # Run batch experiment for HRV error analysis with 5-minute simulations per experiment.
    batch_experiment(debug=True)  # use defaults defined above
    
if __name__ == "__main__":
    main()
