from numpy.fft import rfft, irfft, rfftfreq
import numpy as np
import os
from datetime import datetime

# Additional scientific/plotting imports for curve-fitting pipeline
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

rng = np.random.default_rng(42)

# ai! instead of using pipeline ids, let's use the following map:: {1: combined, 2: missing, 3: jitter, 4: extraneous events
# ai! for any following comments referencing a rename or modify of pipeline ids, please refer to the above map- goal is to get plots to show current pipelines as titles and refer to pipelines by pipeline names rather than idx

# ---------------------------------------------------------------------
# High-level curve-fitting helpers (dataset assembly, models, fitting)
# ---------------------------------------------------------------------

DEBUG = False

def miss_model(hr, q_over_p, T):
    """Pipeline-2 miss-only curve."""
    hr = np.asarray(hr, dtype=float)
    return q_over_p * np.sqrt(T * 60.0 / hr)


def jitter_model(hr, sigma_t, T):
    """Pipeline-3 jitter-only curve."""
    hr = np.asarray(hr, dtype=float)
    return 1.596 * sigma_t * np.sqrt(T) * np.sqrt(hr / 60.0)


def extra_model(hr, lambda_bg, T):
    """Pipeline-4 extraneous-only curve."""
    hr = np.asarray(hr, dtype=float)
    return 0.5 * lambda_bg * np.sqrt(T) * (60.0 / hr) ** 1.5


def combined_model(hr, q_over_p, sigma_t, lambda_bg, T):
    """Pipeline-1 combined curve."""
    return (
        miss_model(hr, q_over_p, T)
        + jitter_model(hr, sigma_t, T)
        + extra_model(hr, lambda_bg, T)
    )


def assemble_hrv_dataset(
    heart_rates=(50, 55, 60, 65, 70, 75, 80),
    n_runs=300,
    T=900,
    jitter_std=0.02,
    p_detect=0.9,
    bg_rate=0.2,
):
    """
    Run simulations and return a pandas.DataFrame with per-pipeline √N-normalised
    MAE values for each heart-rate/run.
    """
    records = []
    for run in range(n_runs):
        local_rng = np.random.default_rng(run)
        for hr in heart_rates:
            period = 60.0 / hr
            n_int = max(int(T / period) - 1, 1)
            scale = np.sqrt(n_int)
            errs = simulate_errors_for_rate(
                period,
                T,
                local_rng,
                jitter_std=jitter_std,
                p_detect=p_detect,
                bg_rate=bg_rate,
                debug=DEBUG,
            )
            for pipeline_id, err in enumerate(errs, start=1):  # ai! replace pipeline_id to be according to map
                records.append(
                    dict(
                        run=run,
                        heart_rate=hr,
                        pipeline=pipeline_id,
                        mae_sqrtN=err * scale,
                    )
                )
    return pd.DataFrame.from_records(records)


# ------------------------- fitting helpers -------------------------


def _fit_curve(func, hr, y, p0):
    popt, _ = curve_fit(func, hr, y, p0=p0, maxfev=10000)
    return popt


def fit_pipeline2(df, T):
    hr = df["heart_rate"].values
    y = df["mae_sqrtN"].values
    return _fit_curve(lambda h, q: miss_model(h, q, T), hr, y, p0=(0.1,))[0]


def fit_pipeline3(df, T):
    hr = df["heart_rate"].values
    y = df["mae_sqrtN"].values
    return _fit_curve(lambda h, s: jitter_model(h, s, T), hr, y, p0=(0.01,))[0]


def fit_pipeline4(df, T):
    hr = df["heart_rate"].values
    y = df["mae_sqrtN"].values
    return _fit_curve(lambda h, l: extra_model(h, l, T), hr, y, p0=(0.01,))[0]


def fit_pipeline1(df, T, huber_delta=1.0):
    hr = df["heart_rate"].values
    y = df["mae_sqrtN"].values
    x0 = (0.1, 0.01, 0.01)

    def residuals(p):
        return combined_model(hr, p[0], p[1], p[2], T) - y

    res = least_squares(
        residuals, x0, loss="huber", f_scale=huber_delta, max_nfev=10000
    )
    return res.x


# ------------------------ bootstrap helper -------------------------


def bootstrap_parameters(fit_func, df, T, n_boot=1000, random_state=0):
    rng_bs = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_boot):
        boot_df = df.sample(len(df), replace=True, random_state=rng_bs.integers(1e9))
        est = fit_func(boot_df, T)
        est = np.atleast_1d(est)
        samples.append(est)
    samples = np.vstack(samples)
    ci_low = np.percentile(samples, 2.5, axis=0)
    ci_high = np.percentile(samples, 97.5, axis=0)
    return samples, ci_low, ci_high


# -------------------------- diagnostics ----------------------------


def _plot_pipeline(df, T, pipeline_id, params, out_dir):
    hr_grid = np.linspace(df.heart_rate.min(), df.heart_rate.max(), 300)
    if pipeline_id == 1:  # ai! rename pipeline id according to map
        pred = combined_model(hr_grid, *params, T)
    elif pipeline_id == 2:
        pred = miss_model(hr_grid, params[0], T)
    elif pipeline_id == 3:
        pred = jitter_model(hr_grid, params[0], T) 
    else:
        pred = extra_model(hr_grid, params[0], T)

    sns.scatterplot(
        data=df, x="heart_rate", y="mae_sqrtN", alpha=0.4, label="simulated"
    )
    plt.plot(hr_grid, pred, color="k", label="fitted")
    plt.title(f"Pipeline {pipeline_id}")
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("MAE · √N  (s)")
    plt.legend()
    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(out_dir) / f"pipeline{pipeline_id}_fit.png", dpi=150)
    plt.clf()


def plot_diagnostics(df, results, T, out_dir="figures"):
    for pid in (1, 2, 3, 4):
        res = results[f"pipeline{pid}"]
        _plot_pipeline(df[df.pipeline == pid], T, pid, list(res.values()), out_dir)

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

def apply_extraneous_events(times, T, bg_rate=0.2, rng=rng):
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
    final_times = apply_extraneous_events(jittered, T, bg_rate, rng)
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
    noisy_extraneous = apply_extraneous_events(ideal, T, bg_rate, rng)
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
                p_detect=p_detect, bg_rate=bg_rate, debug=DEBUG
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

            # accumulate raw errors
            sum_full_raw[i] += err_full_raw
            sum_sq_full_raw[i] += err_full_raw ** 2
            sum_miss_raw[i] += err_miss_raw
            sum_sq_miss_raw[i] += err_miss_raw ** 2
            sum_jitter_raw[i] += err_jitter_raw
            sum_sq_jitter_raw[i] += err_jitter_raw ** 2
            sum_extraneous_raw[i] += err_extraneous_raw
            sum_sq_extraneous_raw[i] += err_extraneous_raw ** 2
            # accumulate normalised errors
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
    parser = argparse.ArgumentParser(
        description="HRV error simulation / curve-fitting utility"
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- batch ---
    p_batch = sub.add_parser("batch", help="run batch experiment and write CSV")
    p_batch.add_argument("--out", default="hrv_error_batch.csv")

    # --- fit ---
    p_fit = sub.add_parser("fit_curves", help="assemble dataset and fit error curves")
    p_fit.add_argument("--out", default="results.json")
    p_fit.add_argument("--figures", default="figures")
    for p in (p_batch, p_fit):
        p.add_argument("--jitter_std", type=float, default=0.02)
        p.add_argument("--p_detect",   type=float, default=0.9)
        p.add_argument("--bg_rate",    type=float, default=0.2)

    args = parser.parse_args()

    if args.cmd == "fit_curves":
        df = assemble_hrv_dataset(jitter_std=args.jitter_std, p_detect=args.p_detect, bg_rate=args.bg_detect)
        T = 900

        q_over_p = fit_pipeline2(df[df.pipeline == 2], T)  # ai! rename pipeline ids according to map: {1: combined, 2: missing, 3: jitter, 4: extraneous events
        sigma_t = fit_pipeline3(df[df.pipeline == 3], T)
        lambda_bg = fit_pipeline4(df[df.pipeline == 4], T)
        q_over_p1, sigma_t1, lambda_bg1 = fit_pipeline1(df[df.pipeline == 1], T)

        results = {  # ai! renae pipeline ids according to above map
            "pipeline1": dict(q_over_p=q_over_p1, sigma_t=sigma_t1, lambda_bg=lambda_bg1),
            "pipeline2": dict(q_over_p=q_over_p),
            "pipeline3": dict(sigma_t=sigma_t),
            "pipeline4": dict(lambda_bg=lambda_bg),
        }

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

        plot_diagnostics(df, results, T, args.figures)
        print(f"[fit_curves] Parameter estimates written to {args.out}")
    else:
        # default to batch experiment
        out_csv = getattr(args, "out", "hrv_error_batch.csv")
        batch_experiment(out_csv=out_csv, debug=True)
    
if __name__ == "__main__":
    main()
