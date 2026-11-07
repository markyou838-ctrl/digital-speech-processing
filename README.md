# YesNo Speech Classification with torchaudio
**Student** — Muntadher Sadoon Kadhim  
**Student ID** — 404131088  
**Course** — Digital Speech Processing


This project trains and compares several classifiers on the YesNo dataset from `torchaudio`. Each recording contains eight spoken words, and each word is either "yes" or "no". The goal is to predict the full sequence of eight words from the audio.

The project compares five classical machine learning models with a one-dimensional convolutional neural network (CNN) implemented in PyTorch. The main finding is that the CNN, which keeps the temporal order of the audio, performs best, and that removing the time dimension causes a severe drop in sequence accuracy.

## Task

Each recording is a sequence of eight words. The target for each recording is an 8-bit vector, where 1 means "yes" and 0 means "no". Because the output is a vector rather than a single label, this is a multi-output classification problem.

Two metrics are reported:

- **Bit accuracy:** the fraction of individual words predicted correctly. A random guess gives about 0.5.
- **Sequence accuracy:** the fraction of recordings where all eight words are predicted correctly. A random guess gives about 1/256, or 0.004.

Sequence accuracy is the stricter metric and better reflects the task, because a single wrong word makes the whole recording wrong.
## Label Encoding

Each word is encoded as a binary value:

| Word | Label |
|---|---|
| no | 0 |
| yes | 1 |

The eight labels of a recording are stored in temporal order, so the label vector gives both the words and their positions. For example, Sample 0 is encoded as:

| Position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Label | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| Word | no | no | no | no | yes | yes | yes | yes |

## Dataset

The dataset is loaded with `torchaudio.datasets.YESNO`. It is downloaded automatically the first time the notebook runs.

| Property | Value |
|---|---|
| Source | `torchaudio.datasets.YESNO` |
| Number of recordings | 60 |
| Sample rate | 8000 Hz |
| Words per recording | 8 |
| Label per word | 0 (no) or 1 (yes) |
| Recording length | Varies between recordings |

Because the recordings differ in length, the features must be converted to a fixed size before they can be used by the classifiers.

## Feature Extraction

Each recording is converted to Mel-frequency cepstral coefficients (MFCCs), which summarize the short-term spectrum of speech in a compact form.

| Parameter | Value |
|---|---|
| Number of MFCC coefficients | 20 |
| FFT size (`n_fft`) | 400 |
| Hop length | 160 samples (20 ms at 8 kHz) |
| Number of Mel bands | 40 |
| Time steps after pooling | 64 |

The MFCC matrix has shape 20 × (number of frames). Adaptive average pooling along the time axis compresses it to 20 × 64. This fixes the input size while keeping the order of the frames, and therefore the order of the words.

The same features are used in two ways:

- **CNN input:** the 20 × 64 matrix, with 20 channels and 64 time steps.
- **Classical model input:** the same matrix flattened into a vector of 1280 values.

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | Standardized inputs, one classifier per output bit |
| SVM | RBF kernel, C = 1, one classifier per output bit |
| K-Nearest Neighbors | k = 3, standardized inputs |
| Random Forest | 300 trees |
| MLP (scikit-learn) | One hidden layer with 128 units, standardized inputs |
| CNN (PyTorch) | Two Conv1d layers (32 and 64 channels, kernel size 3, ReLU, max pooling), a flatten layer, and a linear layer with 8 outputs. Trained with binary cross-entropy loss, Adam optimizer (learning rate 1e-3), batch size 8, and 150 epochs |

The CNN outputs one logit per word, and a sigmoid converts each logit into the probability of "yes". A probability above 0.5 gives label 1.

## Evaluation Protocol

- **Cross-validation:** 5-fold, shuffled.
- **Repetitions:** three random seeds (0, 1, 2) for the fold split.
- **Reported values:** mean ± standard deviation across the runs.
- **Normalization for the CNN:** the mean and standard deviation are computed from the training fold only, to avoid information leaking from the test fold.

## Results

| Classifier | Bit Accuracy | Sequence Accuracy |
|---|---|---|
| **CNN (PyTorch)** | **0.945 ± 0.008** | **0.789 ± 0.031** |
| Logistic Regression | 0.860 ± 0.006 | 0.411 ± 0.021 |
| MLP (scikit-learn) | 0.787 ± 0.003 | 0.128 ± 0.008 |
| Random Forest | 0.740 ± 0.020 | 0.056 ± 0.016 |
| SVM (RBF) | 0.689 ± 0.018 | 0.061 ± 0.016 |
| KNN (k = 3) | 0.674 ± 0.012 | 0.028 ± 0.016 |

The CNN is the best model on both metrics. Its sequence accuracy of 0.789 is almost double that of Logistic Regression, which is the second-best model at 0.411. The gap between the two is much larger than the standard deviations, so it is not an artifact of the random seed.


**The CNN performs best because it preserves temporal order.** Its convolutional filters slide along the time axis, so they detect word patterns in place and keep the order of the frames. Its sequence accuracy of 0.789 shows that it recovers most complete sequences.

**Logistic Regression is a strong baseline.** With standardized MFCC features, a linear decision boundary generalizes well on a small dataset, which explains its second place.

**Temporal structure is essential for this task.** An earlier version of the CNN averaged over all time steps before the classifier. It reached only 0.602 bit accuracy and 0.017 sequence accuracy. Removing the time dimension discarded the order of the words, and the model could not recover it.

**Distance-based and tree-based models struggle here.** KNN and SVM work on distances between 1280-dimensional vectors, which become less informative with few training examples. Random Forest does not model the joint structure of the eight outputs well. These models achieve reasonable bit accuracy but very low sequence accuracy.

## Limitations

- **Small dataset:** 60 recordings is too few for strong claims about generalization. Each training fold contains only about 48 recordings.
- **Architecture changed after seeing results:** the CNN was revised after its first results on the same cross-validation folds. This is a form of selection on the evaluation data, so the CNN's advantage may be somewhat overstated. A cleaner protocol would tune on a separate validation split.
- **Limited hyperparameter tuning:** the settings for each model were chosen by hand, not by a systematic search.
- **Seed coverage:** the CNN's initialization seed is fixed at 0 inside the training function, so its reported standard deviation reflects only the fold split and underestimates the true variability.

## How to Run

1. Open the notebook in Google Colab.
2. Run the cells in order from top to bottom. A CPU runtime is enough. The CNN training takes several minutes on CPU.
3. The dataset is downloaded automatically to `./data`.

To run the notebook locally, install the dependencies:

```bash
pip install torch torchaudio scikit-learn matplotlib python-docx
```

## Files

- `Untitled80 (1).ipynb`: the notebook with the full code, from data loading to results. Rename it to a clearer name such as `YesNo_Classifiers.ipynb` if you prefer.
- `YesNo_Classifiers_Report.docx`: the full written report, including the tables and figures.
- `sample_0.png`, `sample_1.png`, `sample_2.png`, `sample_3.png`: the waveform and spectrogram figures shown above.
