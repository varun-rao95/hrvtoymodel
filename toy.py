from numpy.fft import rfft, irfft, rfftfreq

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
    pass # ai! please fill in this function

def apply_missed_detections(times, p_detect=0.9, rng=rng):
    """
    Keep each event independently with probability p_detect.
    """
    pass # ai! please fill in this function to miss events with prob p

def add_extraneous_events(times, T, bg_rate=0.2, rng=rng):
    """
    Superpose independent homogeneous Poisson 'clutter' with constant rate.
    Returns merged and sorted array.
    """
    pass # AI! please fill in this function

def main():
    pass

if __name__ == "__main__":
    main()
