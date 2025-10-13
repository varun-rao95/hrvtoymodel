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
    return add_extraneous_events(jittered, T, bg_rate, rng) # ai? do we have to ensure within [0, T)

def main():
    import matplotlib.pyplot as plt
    import csv

    T = 60  # simulate for 60 seconds (1 minute)
    period = 1.0  # ideal period of 1 second (~60 bpm)

    # --- Periodic Process ---
    ideal_periodic = generate_periodic_times(T, period)
    noisy_periodic = apply_noise_pipeline(ideal_periodic, T, jitter_std=0.02, p_detect=0.9, bg_rate=0.2, rng=rng)

    # --- Nonhomogeneous Process using nhpp_sinusoidal_times ---
    ideal_nhpp = nhpp_sinusoidal_times(T, lambda0=1.0, alpha=0.8, f=1.0, rng=rng)
    noisy_nhpp = apply_noise_pipeline(ideal_nhpp, T, jitter_std=0.02, p_detect=0.9, bg_rate=0.2, rng=rng)

    # Plot the events for visualization.
    plt.figure(figsize=(12, 8))
    plt.eventplot(ideal_periodic, lineoffsets=1, colors='blue')
    plt.eventplot(noisy_periodic, lineoffsets=2, colors='green')
    plt.eventplot(ideal_nhpp, lineoffsets=3, colors='red')
    plt.eventplot(noisy_nhpp, lineoffsets=4, colors='purple')
    plt.yticks([1, 2, 3, 4], ['Periodic Ideal', 'Periodic Noisy', 'NHPP Ideal', 'NHPP Noisy'])
    plt.xlabel('Time (s)')
    plt.title('Heart Rate Simulation: Ideal and Noisy Events')
    plt.tight_layout()
    plt.savefig("heart_rate_events.png")
    plt.show()

    def write_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"])
            for t in data:
                writer.writerow([t])

    write_csv("periodic_ideal.csv", ideal_periodic)
    write_csv("periodic_noisy.csv", noisy_periodic)
    write_csv("nhpp_ideal.csv", ideal_nhpp)
    write_csv("nhpp_noisy.csv", noisy_nhpp)

if __name__ == "__main__":
    main()
