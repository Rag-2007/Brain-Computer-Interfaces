# Brain-Computer-Interfaces-

This repository contains tools and models for processing, cleaning, and classifying Brain-Computer Interface (BCI) EEG data. It is broadly divided into two main components:
1. **SAM40 Dataset Analysis** (Mental Stress vs. Relax classification)
2. **Motor Imagery Classification** (BCI Competition IV 2a)

---

## 1. SAM40 Dataset Analysis

The following scripts are designed to work together to clean, visualize, and classify the SAM40 dataset, which consists of EEG recordings of subjects in "Relax" and "Stress" (Arithmetic) states.

### `eeg_clean.py`
This script provides an automated pipeline for cleaning raw EEG signals to remove artifacts (like eye blinks and muscle noise).
*   **Bandpass Filtering:** Applies a 4th-order Butterworth bandpass filter (0.5 Hz - 40 Hz) to isolate relevant brainwave frequencies.
*   **Artifact Removal via ICA:** Uses FastICA (Independent Component Analysis) to decompose the EEG signal into independent sources. It automatically identifies and zeroes out artifact components based on a **Kurtosis threshold** (> 5.0).
*   **Output:** Reconstructs the cleaned signal and saves it as a new `.mat` file in a `NEW_PY/` directory.

### `compare.py`
A utility script to quantitatively compare the quality of the EEG cleaning process.
*   It loads two `.mat` files—typically a Python-cleaned EEG file (from `eeg_clean.py`) and a MATLAB-cleaned baseline.
*   **Metrics Evaluated:** Calculates Overall Standard Deviation (STD), Mean Channel STD, and Peak Amplitude.
*   **Visualization:** Plots these metrics side-by-side using a Matplotlib bar chart, allowing for easy visual validation of the Python cleaning pipeline against established MATLAB methods.

### `graph_plot.py`
A visualization script that uses the `mne` (MNE-Python) library to plot the multi-channel EEG time-series data.
*   Loads a cleaned `.mat` file and assigns 32 standard 10-20 system channel names (e.g., Fp1, Fz, Cz, O1).
*   Creates an interactive MNE plot for the user to visually inspect the cleaned EEG signals at a sampling frequency of 128 Hz.

### `classify.py`
The core machine learning pipeline for classifying the SAM40 dataset into two states: **Relax (0)** and **Stress (1)**.
*   **Data Loading & Segmentation:** Iterates through `.mat` files, normalizes channels using z-score normalization, and slices the continuous EEG data into overlapping 2-second windows (Sliding-window segmentation).
*   **Model Architecture:** Implements a **Shallow Convolutional Neural Network (ShallowConvNet)** in PyTorch, which is highly effective for oscillatory EEG feature extraction. It uses temporal convolutions followed by spatial convolutions.
*   **Training Techniques:**
    *   **Mixup Augmentation:** Blends samples and labels during training to improve generalization.
    *   **Label Smoothing:** Softens hard labels to prevent overconfidence and overfitting.
    *   **Class Weighting & Weighted Sampling:** Handles any class imbalances between Relax and Stress segments.
    *   **Cosine Annealing:** Smoothly decays the learning rate for better convergence.
*   **Evaluation:** Uses Stratified K-Fold Cross-Validation, reporting Accuracy, Balanced Accuracy, and Macro-F1 score, alongside dynamic threshold tuning.

---

## 2. Motor Imagery Classification

### `EEGNet.ipynb`
This Jupyter Notebook is dedicated to **Motor Imagery Classification** (Left hand, Right hand, Foot, Tongue) using the **BCI Competition IV 2a** dataset. 
*(Note: This is completely separate from the SAM40 dataset analysis).*

*   **Data Extraction & Preprocessing:** 
    *   Downloads the BCI Competition dataset directly from the source.
    *   Uses MNE to load `.gdf` files, drops EOG (eye movement) channels, and applies bandpass (4-40 Hz) and notch filters (50 Hz) to remove line noise.
    *   Extracts 4-second epochs immediately following the presentation of the motor imagery cue.
*   **Model Architecture - EEGNet:**
    *   Implements the renowned **EEGNet (EEGNET-8,2)** model in PyTorch.
    *   EEGNet is a compact Convolutional Neural Network specifically designed for EEG signals. It utilizes **Depthwise Convolutions** (to learn spatial filters) and **Separable Convolutions** (to learn temporal summaries), making it highly parameter-efficient.
*   **Training & Evaluation:**
    *   Splits the data into a 90/10 Train/Test set.
    *   Trains the model using Cross-Entropy Loss and the Adam optimizer over 500 epochs.
    *   Evaluates the model on the test set and generates a confusion matrix to visualize the prediction accuracy across the 4 motor imagery classes.
