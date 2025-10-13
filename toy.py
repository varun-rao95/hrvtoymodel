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

def main():
    import matplotlib.pyplot as plt
    import csv

    T = 60  # simulate for 60 seconds (1 minute)
    period = 1.0  # ideal period of 1 second (~60 bpm)

    # Generate ideal periodic events.
    ideal = generate_periodic_times(T, period)

    # Generate noisy events:
    # 1. Add timestamp jitter.
    jittered = add_timestamp_jitter(ideal, jitter_std=0.02)
    # 2. Apply missed detections.
    missed = apply_missed_detections(ideal, p_detect=0.9)
    # 3. Superpose extraneous events.
    cluttered = add_extraneous_events(ideal, T, bg_rate=0.2)

    # Plot the events for visualization.
    plt.figure(figsize=(10, 6))
    plt.eventplot(ideal, lineoffsets=1, colors='blue')
    plt.eventplot(jittered, lineoffsets=2, colors='green')
    plt.eventplot(missed, lineoffsets=3, colors='red')
    plt.eventplot(cluttered, lineoffsets=4, colors='purple')
    plt.yticks([1, 2, 3, 4], ['Ideal', 'Jittered', 'Missed', 'Cluttered'])
    plt.xlabel('Time (s)')
    plt.title('Periodic Events with Noise (Heart Rate Simulation)')
    plt.tight_layout()
    plt.savefig("events_plot.png")
    plt.show()

    # Write CSV files for each series of events.
    def write_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"])
            for t in data:
                writer.writerow([t])

    write_csv("ideal.csv", ideal)
    write_csv("jittered.csv", jittered)
    write_csv("missed.csv", missed)
    write_csv("cluttered.csv", cluttered)

if __name__ == "__main__":
    main()
