# YesNo Speech Classification with torchaudio
## Muntadher Sadoon Kadhim
## stuent ID :404131088


This project trains and compares several classifiers on the YesNo dataset from `torchaudio`. Each recording contains eight spoken words, and each word is either "yes" or "no". The goal is to predict the full sequence of eight words from the audio.

The project compares five classical machine learning models with a one-dimensional convolutional neural network (CNN) implemented in PyTorch. The main finding is that the CNN, which keeps the temporal order of the audio, performs best, and that removing the time dimension causes a severe drop in sequence accuracy.

## Task

Each recording is a sequence of eight words. The target for each recording is an 8-bit vector, where 1 means "yes" and 0 means "no". Because the output is a vector rather than a single label, this is a multi-output classification problem.

Two metrics are reported:

- **Bit accuracy:** the fraction of individual words predicted correctly. A random guess gives about 0.5.
- **Sequence accuracy:** the fraction of recordings where all eight words are predicted correctly. A random guess gives about 1/256, or 0.004.

Sequence accuracy is the stricter metric and better reflects the task, because a single wrong word makes the whole recording wrong.

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

## Waveforms and Spectrograms

Each figure below shows one recording. The top panel is the waveform, which plots amplitude against time. The bottom panel is the spectrogram, which shows how the frequency content changes over time. The title of each figure lists the eight word labels in temporal order, so the figures can be compared directly with the labels.

### Sample 0

![Sample 0: waveform and spectrogram](sample_0.png)

### Sample 1

![Sample 1: waveform and spectrogram](sample_1.png)

### Sample 2

![Sample 2: waveform and spectrogram](sample_2.png)

### Sample 3

![Sample 3: waveform and spectrogram](sample_3.png)

In the waveform, each spoken word appears as a burst of amplitude, and the silent gaps between words appear as low-amplitude regions. The spectrogram shows the same words as regions of energy at their own time positions. This structure is why the temporal order matters: the classifier has to know not only which words are present, but where each one occurs.

## Discussion

**The CNN performs best because it preserves temporal order.** Its convolutional filters slide along the time axis, so they detect word patterns in place and keep the order of the frames. Its sequence accuracy of 0.789 shows that it recovers most complete sequences.

**Logistic Regression is a strong baseline.** With standardized MFCC features, a linear decision boundary generalizes well on a small dataset, which explains its second place.

**Temporal structure is essential for this task.** An earlier version of the CNN averaged over all time steps before the classifier. It reached only 0.602 bit accuracy and 0.017 sequence accuracy. Removing the time dimension discarded the order of the words, and the model could not recover it.

**Distance-based and tree-based models struggle here.** KNN and SVM work on distances between 1280-dimensional vectors, which become less informative with few training examples. Random Forest does not model the joint structure of the eight outputs well. These models achieve reasonable bit accuracy but very low sequence accuracy.

## Limitations

- **Small dataset:** 60 recordings is too few for strong claims about generalization. Each training fold contains only about 48 recordings.
- **Architecture changed after seeing results:** the CNN was revised after its first results on the same cross-validation folds. This is a form of selection on the evaluation data, so the CNN's advantage may be somewhat overstated. A cleaner protocol would tune on a separate validation split.
- **Limited hyperparameter tuning:** the settings for each model were chosen by hand, not by a systematic search.
- **Seed coverage:** the CNN's initialization seed is fixed at 0 inside the training function, so its reported standard deviation reflects only the fold split and underestimates the true variability.



- # 2- Speech Analysis on the VOiCES Subset

Acoustic event classification and voice activity detection (VAD) on the VOiCES subset.

📄 **Full report (Word):** [VOiCES_Event_and_VAD_Report.docx](VOiCES_Event_and_VAD_Report.docx)

---

## 1. Introduction

This project applies two speech processing tasks to the VOiCES subset. VOiCES (Voices Obscured in Complex Environmental Settings) contains speech recorded in real reverberant rooms. The foreground speaker is replayed LibriSpeech, and the distractor noise is taken from MUSAN. This subset contains recordings where the main speaker is located at 90 degrees.

The two tasks are:

- **Task A: Acoustic event classification.** Identify the background noise type of each recording.
- **Task B: Voice activity detection (VAD).** Label each time frame as speech or non-speech.

## 2. Data

The reference file lists 1,500 recordings with their transcripts. The archives distributed in the repository (`VOiCES_90_1.tar` to `VOiCES_90_8.tar`) contain 908 of these recordings, so all experiments use those 908 files. The 592 missing recordings are not present in the archives, and the reason is not known.

Every recording is mono, sampled at 16 kHz, and lasts about 16 seconds.

| Property | Value |
|---|---|
| Recordings used | 908 of 1,500 |
| Noise types | babble 228, music 231, none 227, TV 222 |
| Rooms | rm1: 446, rm2: 462 |
| Speakers | 104 |
| Chapters (groups for validation) | 122 |
| Sample rate / channels | 16,000 Hz / mono |

Each filename encodes the room, noise type, speaker, and chapter. The noise labels for Task A are taken from the filename. Macintosh metadata files (prefixed with `._`) were removed.

## 3. Task A: Acoustic Event Classification

The subset has no time-stamped event annotations. Event detection is therefore formulated as **clip-level classification** of the background noise type into four classes: babble, music, none, and TV. This is a simplification of frame-level event detection.

### 3.1 Features

Each recording is converted to 20 MFCC coefficients (`n_fft = 512`, hop length = 160 samples, 40 mel bands). The clip-level feature vector is the mean and standard deviation of each coefficient over time, giving **40 features per recording**.

### 3.2 Models and Evaluation

| Model | Configuration |
|---|---|
| Logistic Regression | Standardized inputs, `max_iter = 3000` |
| SVM | RBF kernel, C = 1, standardized inputs |
| Random Forest | 300 trees, `random_state = 0` |
| MLP (PyTorch) | 40-64-4, ReLU, dropout 0.2, Adam (lr = 1e-3), cross-entropy, 200 full-batch epochs |

Evaluation uses **5-fold GroupKFold with the chapter as the group**, so no chapter (and therefore no sentence) appears in both training and test data. Accuracy and macro-averaged F1 are reported as mean ± standard deviation across folds.

### 3.3 Results

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| **SVM (RBF)** | **0.754 ± 0.027** | **0.749 ± 0.030** |
| MLP (PyTorch) | 0.730 ± 0.013 | 0.725 ± 0.010 |
| Random Forest | 0.703 ± 0.024 | 0.691 ± 0.029 |
| Logistic Regression | 0.657 ± 0.024 | 0.652 ± 0.023 |

All classifiers perform well above chance (0.25 for four balanced classes). SVM achieves the highest accuracy. The MLP is close to SVM, which is expected given the small dataset of 908 recordings.

### 3.4 Confusion Analysis

The figure below shows the SVM confusion matrix, pooled over all out-of-fold predictions. Per-class recall: **babble 0.860, music 0.818, none 0.771, TV 0.563**.

<p align="center">
  <img src="event_confusion.png" width="450" alt="SVM confusion matrix for noise type classification">
</p>

*Figure 1. Event classification (noise type): SVM confusion matrix, pooled over out-of-fold predictions.*

The most frequent confusion is between **TV and none** (43 and 27 recordings), followed by **TV and babble** (27 and 23). TV is the hardest class. A plausible explanation is that recordings labeled `none` still contain room noise and HVAC hum, which resemble low-level TV audio. The matrix alone does not verify this. Because the features are clip-level statistics, they discard temporal detail, which may limit the separation of quiet noise types.

## 4. Task B: Voice Activity Detection

### 4.1 Reference Labels

The subset has no manual speech annotations. Frame labels were generated by **forced alignment**: the transcript was normalized (uppercase, punctuation removed) and aligned to the audio using the CTC acoustic model `WAV2VEC2_ASR_BASE_960H` from torchaudio. Each word spans from its first to its last aligned character, and the frames inside these spans are labeled as speech. A margin of two frames (**40 ms**) is added on both sides of each word. Frames are 20 ms long.

Across the 908 recordings there are **705,867 frames**, of which **66.6% are labeled speech**.

The margin was chosen by inspecting the resulting speech ratio, not by optimizing a criterion. A 100 ms margin merged almost all pauses (92% speech on the example recording) and was rejected. A 40 ms margin gives about 76% speech on the same recording, which was judged closer to the speech actually present.

### 4.2 Methods

- **Baseline (energy-based VAD):** the energy of each frame is computed in decibels. A frame is labeled speech when its energy exceeds the 10th-percentile noise floor of the recording by 10 dB. The threshold is fixed and was not tuned against the labels.
- **Learned VAD:** Logistic Regression on MFCC features (20 coefficients, `n_fft = 512`, hop length = 320 samples, interpolated to the 20 ms label rate). Each frame is represented by its own features and those of two frames before and after it, giving 100 features. Features are standardized on the training folds, and at most 100,000 frames are sampled per fold for training.

Both methods are evaluated on the same 5-fold GroupKFold split by chapter, so the comparison is fair.

### 4.3 Results

| Method | Frame Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** (MFCC, ±2 frames) | **0.829 ± 0.002** | 0.836 ± 0.017 | **0.925 ± 0.020** | **0.878** |
| Energy baseline | 0.753 ± 0.010 | 0.851 ± 0.011 | 0.762 ± 0.020 | 0.804 ± 0.008 |

Logistic Regression outperforms the energy baseline in accuracy and F1. The main difference is **recall**: the energy method misses quieter speech, which is expected when babble, music, or TV raises the noise floor. The energy baseline has slightly higher precision, meaning fewer false speech detections, but at the cost of many missed speech frames.

The figure below shows one example recording per noise type. The orange regions are the VAD reference, and the spectrogram is on the right.

<p align="center">
  <img src="vad_examples.png" width="850" alt="VAD reference and spectrograms for four noise types">
</p>

*Figure 2. Examples of recordings (one per noise type). Orange regions show the VAD reference; the spectrogram is shown on the right.*

## 5. Limitations

- **Partial data:** 908 of the 1,500 recordings in the reference file are available in the archives.
- **Silver labels:** the VAD reference comes from forced alignment, not human annotation. The VAD results measure agreement with the alignment-based reference, not absolute speech detection accuracy.
- **Simplified event detection:** Task A classifies clip-level noise type, not time-localized events, because the subset has no event timestamps.
- **Scope:** a small dataset and a single microphone position (90 degrees), which limits generalization.
- **Tuning:** hyperparameters were not tuned systematically, and the 40 ms margin was chosen by inspection.

## 6. Conclusion

On the VOiCES subset, noise type can be recognized from clip-level MFCC statistics, with SVM reaching 0.754 accuracy. For voice activity detection, a Logistic Regression model using short-context MFCC features outperforms an energy-based baseline (F1 0.878 against 0.804), mainly by detecting more speech.

Both conclusions depend on the alignment-based labels and the 908 available recordings. Future work should collect manual speech annotations, add time-stamped event labels, and tune the models on a separate validation split.

## Files

- `VOiCES_Event_and_VAD_Report.docx`: the full report with tables and figures
- `event_confusion.png`: Figure 1, SVM confusion matrix
- `vad_examples.png`: Figure 2, VAD examples


