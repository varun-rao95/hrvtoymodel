6. **Ablations**

   * Set $\sigma_t=0$ → right tail flattens.
   * Set $\lambda_{\mathrm{bg}}=0$ → low-HR slope weakens but persists (miss term).

---

## 2) Repeat on **real PPG** (public datasets)

**Goal:** reproduce HR-dependent error curves using PPG vs ECG ground truth.

### Suggested datasets

* **PPG-DaLiA** — wrist PPG + chest ECG with ground-truth HR; daily-life activities (strong motion artefacts).
  [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)
* **WESAD** — wrist BVP (PPG) + chest ECG (lab stress protocol); clean lab conditions, rich modalities.
  [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/WESAD%2B%28Wearable%2BStress%2Band%2BAffect%2BDetection%29)
* **BIDMC PPG & Respiration (PhysioNet)** — ICU PPG; pair with ECG in MIMIC for clinical noise patterns.
  [PhysioNet](https://physionet.org/content/bidmc/)

*(For scale, MIMIC waveform DB has lots of PPG+ECG but is heavier to parse.)*
[PhysioNet Waveforms](https://physionet.org/content/?topic=waveform)

### Pipeline on real data

1. **Acquire & subset**
   Start with a few subjects/sessions (avoid huge pulls). PPG-DaLiA/WESAD are easy to download.

2. **Preprocess**
   Bandpass PPG (e.g., 0.5–8 Hz). Optionally apply a motion mask (accel magnitude threshold).

3. **Beat detection (PPG)**
   Simple peak-finding after derivative/squaring with a refractory period and min peak distance (e.g., 0.3–1.5 s).
   Helpful overview: [Detecting beats in the photoplethysmogram (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9393905/)

4. **Ground truth (ECG)**
   Detect R-peaks (Pan–Tompkins variant) or use provided annotations; derive ECG IBIs.

5. **Align & pair**
   Sync clocks; if needed, cross-correlate HR series to remove constant offsets.
   For each PPG IBI, take the matching ECG R–R as truth.

6. **Compute errors & stratify**
   Per interval: $E = \mathrm{IBI}*{\mathrm{PPG}} - \mathrm{IBI}*{\mathrm{ECG}}$, use $|E|$.
   Rolling HR $= 60/\mathrm{IBI}*{\mathrm{ECG}}$.
   Bin by HR (e.g., 5-bpm bins). For each bin: MAE and $\mathrm{MAE}*{\sqrt N}$.

7. **Fit the same formulas**
   Treat detector misses/extras as effective $q/p$ and $\lambda_{\mathrm{bg}}$; $\sigma_t$ is effective timestamp noise.
   Run NLS fits; overlay curves.

8. **Robustness slices**

   * Activity (rest vs motion) — PPG-DaLiA has activity labels.
   * Device position (wrist vs chest, when available).
   * Clinical vs non-clinical (BIDMC/MIMIC vs WESAD/DaLiA).

9. **Report**
   Simulated vs real curves, parameter estimates with CIs, where the jitter term bends the right tail, and where extras dominate (low HR).

---

## Quick checkpoints to avoid traps

* **Window length $T$**: keep $T$ fixed across HR bins so $\sqrt{N}$ normalisation is apples-to-apples.
* **Miss accounting**: on real data, missed PPG beats → unusually long IBIs. Flag any IBI $> 1.8\times$ local median for QA.
* **Extras**: unusually short IBIs ($< 0.5\times$ local median) suggest spurious peaks—analyze with/without them.

---

## Nice-to-have utilities (fast follow-ups)

* Export per-bin summaries: HR bin, $N$, MAE, $\mathrm{MAE}_{\sqrt N}$, CI.
* A predictor that plots the curve for chosen $(q/p,\ \sigma_t,\ \lambda_{\mathrm{bg}})$ (e.g., “what if $\sigma_t=20$ ms?”).
* A notebook cell that fits both pipelines and prints a tidy table of parameters with bootstrap CIs.

---

