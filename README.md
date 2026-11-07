# YesNo Speech Classification with torchaudio
**Student** — Muntadher Sadoon Kadhim  
**Student ID** — 404131088  
**Course** — Digital Speech Processing
This project compares classical machine learning models and a 1D CNN on the YesNo dataset from `torchaudio`. The goal is to recognize the sequence of eight spoken words, where each word is either "yes" or "no".

## Table of Contents

1. [Task](#task)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Models](#models)
5. [Results](#results)
6. [Waveforms and Spectrograms](#waveforms-and-spectrograms)
7. [Discussion](#discussion)
8. [Limitations](#limitations)
9. [How to Run](#how-to-run)
10. [Files](#files)

## Task

Each recording contains eight words. The target for each recording is an 8-bit vector, where 1 means "yes" and 0 means "no". Two metrics are reported:

- **Bit accuracy:** the fraction of correctly predicted words.
- **Sequence accuracy:** the fraction of recordings where all eight words are correct.

## Dataset

| Property | Value |
|---|---|
| Source | `torchaudio.datasets.YESNO` |
| Recordings | 60 |
| Sample rate | 8000 Hz |
| Words per recording | 8 |
| Labels per word | 0 (no) or 1 (yes) |

## Features

Each recording is converted to MFCC features with the following settings:

| Parameter | Value |
|---|---|
| Number of MFCC coefficients | 20 |
| FFT size (`n_fft`) | 400 |
| Hop length | 160 |
| Mel bands | 40 |
| Time steps after pooling | 64 |

Recordings vary in length, so the frames are pooled to 64 time steps with adaptive average pooling. This gives a fixed input size while keeping the temporal order of the words.

- **CNN input:** 20 × 64 matrix
- **Classical models input:** the same features flattened to 1280 values

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | Standardized inputs, one classifier per bit |
| SVM | RBF kernel, C = 1, one classifier per bit |
| KNN | k = 3, standardized inputs |
| Random Forest | 300 trees |
| MLP | One hidden layer with 128 units |
| CNN (PyTorch) | Two Conv1d layers (32 and 64 channels), max pooling, linear output with 8 logits, BCE loss, Adam (lr = 1e-3), 150 epochs |

## Results

Evaluation uses 5-fold cross-validation, repeated with three random seeds. Values are reported as mean ± standard deviation.

| Classifier | Bit Accuracy | Sequence Accuracy |
|---|---|---|
| **CNN (PyTorch)** | **0.945 ± 0.008** | **0.789 ± 0.031** |
| Logistic Regression | 0.860 ± 0.006 | 0.411 ± 0.021 |
| MLP (sklearn) | 0.787 ± 0.003 | 0.128 ± 0.008 |
| Random Forest | 0.740 ± 0.020 | 0.056 ± 0.016 |
| SVM (RBF) | 0.689 ± 0.018 | 0.061 ± 0.016 |
| KNN (k = 3) | 0.674 ± 0.012 | 0.028 ± 0.016 |

## Waveforms and Spectrograms

Each figure shows the waveform (top) and spectrogram (bottom) of one recording. The title lists the eight word labels in temporal order.

![Sample 0](sample_0.png)

![Sample 1](sample_1.png)

![Sample 2](sample_2.png)

![Sample 3](sample_3.png)

| Figure | Recording | What to observe |
|---|---|---|
| Sample 0 | Index 0 | Eight bursts of energy, one per word, separated by silence |
| Sample 1 | Index 1 | Different word pattern with the same structure |
| Sample 2 | Index 2 | Word timing varies between recordings |
| Sample 3 | Index 3 | Energy concentrated in the lower frequency bands for speech |

In the spectrogram, each word appears as a region of energy at its own time position, which helps explain why the temporal structure matters for classification.

## Discussion

- **The CNN performs best.** Its convolutional layers operate along the time axis, so they detect word patterns while preserving their order.
- **Logistic Regression is a strong baseline.** With standardized MFCC features, a linear decision boundary generalizes well on a small dataset.
- **Temporal structure matters.** An earlier CNN that averaged over all time steps reached only 0.602 bit accuracy and 0.017 sequence accuracy. Removing the time dimension discards word order.

## Limitations

- **Small dataset:** 60 recordings is too few for strong generalization claims.
- **Architecture changed after seeing results:** the CNN was revised after its first results on the same folds, which may overstate its advantage. A clean protocol would tune on a separate validation split.
- **Limited tuning:** hyperparameters were not searched systematically.
- **Seed coverage:** the CNN's initialization seed is fixed, so its standard deviation underestimates true variability.

## How to Run

1. Open the notebook in Google Colab.
2. Runtime → Run all (CPU is enough; GPU is optional).
3. The dataset downloads automatically to `./data`.

Install the dependencies if needed:

```bash
pip install torch torchaudio scikit-learn matplotlib python-docx
```

## Files

- `Untitled80 (1).ipynb`: the notebook with the full code (rename it to a clearer name if you prefer)
- `YesNo_Classifiers_Report.docx`: the full report with figures and tables
- `sample_0.png` to `sample_3.png`: waveform and spectrogram figures
