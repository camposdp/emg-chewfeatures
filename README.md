# 📃 EMG Feature Extraction and Classification Pipeline

This repository contains a **full pipeline** for:

- Extracting EMG features from segmented chewing signals.
- Selecting the best feature combinations via forward feature selection.
- Evaluating classification performance using different machine learning models.
- Generating publication-ready plots and tables.

Designed for easy integration into Overleaf and GitHub.

---

# 🔧 Repository Structure

```plaintext
├── raw_data/                  % Raw .mat files for each chewing repetition
│    ├── Capim Estrela Africana/
│    ├── Capim Kurumi/
│    └── Feno/
├── extract_features.m         % Step 1: Preprocessing, segmentation, and feature extraction
├── select_features.m          % Step 2: Forward feature selection
├── evaluate_models.m          % Step 3: Model evaluation, figure generation, statistical testing
├── run_pipeline.m             % Run all steps sequentially
├── figures/                   % (Auto) Plots in high-quality PDF (colorblind friendly)
├── tables/                    % (Auto) Tables summarizing results (.xlsx)
├── README.md                  % This file
```

---

# 🔢 Methodology

## 1. Feature Extraction

- Signals are first **high-pass filtered** at **20 Hz** (Butterworth 4th order) to remove baseline drift.
- Envelope extraction via **moving RMS window** (50 samples = 50 ms).
- Segmentation is done by **thresholding**:
  - Threshold = mean + 3 * std of baseline.
  - Smoothing of the binary signal (window = 50 samples).
- Segments (bursts) are detected individually for:
  - Left channel (CH1)
  - Right channel (CH2)
  - Combined signal (average of envelopes).

For each valid segment, the following **9 features** are computed:

| Feature | Description |
|:--------|:------------|
| RMS     | Root Mean Square |
| MAV     | Mean Absolute Value |
| ZC      | Zero Crossings |
| WL      | Waveform Length |
| SSC     | Slope Sign Changes |
| IEMG    | Integrated EMG (area) |
| BD      | Burst Duration |
| IchT    | Inter-contraction Time |
| CD      | Cycle Duration |

Separate feature sets are built for Left, Right, and Both channels.

---

## 2. Feature Selection (Forward Selection)

- **5-fold cross-validation** is used to evaluate feature combinations.
- In the case of "Both" sides, feature groups are paired (e.g., RMS_Left + RMS_Right).
- At each iteration, the feature (or pair) that most improves the mean cross-validation accuracy is added.
- The accuracy progression is saved and plotted.

---

## 3. Model Evaluation

The following classifiers are evaluated:

- **SVM (RBF Kernel)**
- **SVM (Linear Kernel)**
- **LDA (Linear Discriminant Analysis)**
- **KNN (k-Nearest Neighbors)**

**Best feature combination** found for each side and method is used.

Outputs:
- Accuracy progression as features are added.
- Final 5-fold accuracies (mean and std).
- Confusion matrices.
- Statistical comparison between Left, Right, and Both sides (ANOVA).

---

# 📊 Main Parameters Used

| Parameter | Value | Notes |
|:----------|:------|:------|
| High-pass filter cutoff | 20 Hz | 4th order Butterworth |
| Moving RMS window | 50 samples | Envelope extraction |
| Segmentation threshold | mean + 3*std | Over smoothed RMS |
| Smoothing window for binary mask | 50 samples | Post-threshold |
| Feature extraction per segment | 9 features | Detailed above |
| Classifiers | SVM (RBF), SVM (Linear), LDA, KNN | |
| Cross-validation | 5-fold | Stratified |

---

# 📉 Outputs

## Figures:
- Forward feature selection curves (Accuracy vs Features Added)
- Final accuracies (Bar plots ± standard deviation)
- Confusion matrices (normalized)

## Tables:
- Selected features per method and side.
- Accuracy progression across selected features.
- Statistical comparison between sides.

_All plots are saved in vector-based PDF format (Overleaf-ready) with correct page size and colorblind-friendly colors._

---

# 👩‍💻 How to Run

In MATLAB:

```matlab
run('run_pipeline.m');
```

All steps will execute sequentially.

---

# 💡 Notes

- Figures are designed for direct insertion into LaTeX (no manual adjustments).
- Statistical significance (α = 0.05) is printed and saved in the tables.
- Feature names, legends, and titles are all in English.
- Color schemes suitable for colorblind readers.

---

# 👨‍💻 Authors

- Data Processing: Daniel Campos
- Signal Segmentation: Gabriel Harres
- Machine Learning and Evaluation: Daniel Campos


---

# ☑️ Cite this work



---

