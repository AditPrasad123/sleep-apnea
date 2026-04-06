# 🧠 1. LOADING THE DATA

### What we do:

* We read ECG recordings using `wfdb`

### What data is there:

* Each file contains:

  * ECG signal (continuous waveform)
  * Annotation file (`apn`) → labels per minute

### Data type:

* Signal → `float array`
* Labels → `list of characters ('A' or 'N')`

---

# 🧠 2. SEGMENTATION (VERY IMPORTANT STEP)

### What we do:

```text
Loop over signal → cut into 30-second windows
```

* Sampling rate = 100 Hz
* 30 seconds → **3000 samples**
* Stride = 10 seconds → overlapping windows

---

### What data becomes:

Before:

```text
1 long signal → e.g. 1 hour
```

After:

```text
many segments → each of shape (3000,)
```

---

### Label assignment:

```text
label_idx = start // window_size
```

* Each segment gets a label:

  * `1` → apnea
  * `0` → normal

---

### Data type now:

```text
X_signal → (N, 3000) float
y → (N,) int
```

---

# 🧠 3. NORMALIZATION + CLEANING

### What we do:

```python
segment = (segment - mean) / std
```

Then:

* remove NaN / Inf
* skip bad segments

---

### Why:

* ensures stable training
* removes corrupted data

---

# 🧠 4. FEATURE EXTRACTION (IMPORTANT FOR XAI + XGBOOST)

For each segment → we compute features

---

## 🔹 4.1 Time-domain features

From raw signal:

* mean
* std
* variance
* RMS
* max / min
* median
* percentiles

👉 describe **signal amplitude + distribution**

---

## 🔹 4.2 Peak detection

```python
peaks = find_peaks(...)
```

From this → RR intervals

---

## 🔹 4.3 HRV features

From RR intervals:

* heart rate
* mean RR
* std RR
* RMSSD
* pNN50

Extra:

* max RR, min RR
* quartiles

👉 describe **heart rhythm variability**

---

## 🔹 4.4 Signal complexity

* energy → signal strength
* entropy → randomness

---

## 🔹 4.5 Frequency features

Using FFT:

* LF power
* HF power
* LF/HF ratio

👉 describe **autonomic nervous system behavior**

---

## 🔹 4.6 Statistical shape

* skewness
* kurtosis
* zero-crossing rate

---

### Final feature vector:

```text
X_features → (N, ~25–30)
Type: float
```

---

# 🧠 5. FEATURE SCALING

```python
StandardScaler()
```

👉 converts features to:

* mean = 0
* std = 1

---

# 🧠 6. TRAIN-TEST SPLIT

### What we do:

```text
Split indices → apply same split everywhere
```

---

### Why:

* ensures consistency between:

  * XGBoost
  * CNN
  * stacking

---

# 🧠 7. MODEL 1 — XGBOOST

### Input:

```text
X_features
```

### What it learns:

* statistical patterns
* HRV relationships

### Output:

```text
probability of apnea
```

---

# 🧠 8. MODEL 2 — CNN-LSTM Hybrid Network

### Input:

```text
X_signal → (3000, 1)
```

---

### What happens:

#### Step 1: Conv1D

* extracts local waveform patterns

#### Step 2: LSTM

* captures temporal dependencies

---

### Output:

```text
probability of apnea
```

---

# 🧠 9. HYBRID FEATURE FUSION

Inside CNN model:

```text
[ECG features] + [engineered features]
```

👉 combined → dense layers → prediction

---

# 🧠 10. STACKING (FINAL MODEL)

### Inputs:

```text
[XGBoost prob, CNN prob]
```

### Model:

The meta-model learns to optimally combine predictions from both models.

```text
XGBClassifier
```

---

### What it does:

* learns which model to trust more

---

# 🧠 FINAL DATA FLOW (VERY IMPORTANT)

```text
Raw ECG → segmentation → normalized segments

→ Path 1: Feature extraction → XGBoost
→ Path 2: Raw signal → CNN + LSTM

→ Combine predictions → stacking → final output

Feature-level fusion is performed within the deep learning model, while decision-level fusion is achieved through stacking.
```

---


> “We segment ECG signals into 30-second windows, extract physiological features, and simultaneously feed raw signals into a CNN-LSTM model. The outputs of both models are combined using a stacking approach to improve prediction accuracy.”

---


> “We combine handcrafted physiological features with deep temporal representations using a hybrid feature learning and stacking-based ensemble approach, improving robustness and interpretability.”

---

> “We integrate handcrafted physiological features with deep temporal representations through a hybrid feature learning framework, followed by stacking-based ensemble learning to enhance robustness and interpretability.”
