# Task 8: Yes/No Audio Classification and Audio Event Detection on VOiCES

**Student** — Muntadher Sadoon Kadhim  
**Student ID** — 404131088  
**Course** — Digital Speech Processing

**[Open Interactive Demo](https://markyou838-ctrl.github.io/digital-speech-processing/task8/index.html)** · [View Notebook](Audio_Assignment.ipynb) · [Browse CSV Results](results/)

## 1. Objective

This assignment has two parts. In the first, I trained and compared three classifiers on the YESNO dataset. In the second, I used the VOiCES subset to demonstrate voice activity detection with Silero VAD and audio event detection with YAMNet.

## 2. YESNO Word Classification

### Data preparation and segmentation

I started with 60 recordings from the torchaudio YESNO dataset. Each filename contains eight labels, where `0` means No and `1` means Yes. To train a word classifier, I first had to segment each recording into individual words and match each segment to its label in order.

I tested the segmentation across all recordings. Some adjacent words were merged into one segment, so I inspected the waveforms and adjusted the segmentation. I corrected one remaining recording manually. For another recording, I heard seven words while the filename provided eight labels. I excluded that recording because I could not reliably match the seven words to the eight labels.

After this check, I kept **59 recordings and 472 word samples**.

### Features and train/test split

For each word, I extracted 13 Mel-frequency cepstral coefficients (MFCCs). I calculated the mean and standard deviation of each coefficient over time, giving **26 features per word**.

I split the data by recording. This kept all words from each recording in the same set.

| Split | Recordings | Words | No | Yes |
| :--- | ---: | ---: | ---: | ---: |
| Training | 47 | 376 | 180 | 196 |
| Test | 12 | 96 | 43 | 53 |
| **Total** | **59** | **472** | **223** | **249** |

### Classifier results

I trained K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest on the same training set and evaluated them on the same test set.

| Classifier | Accuracy | Macro F1 | Correct / Total |
| :--- | ---: | ---: | ---: |
| KNN | 100% | 100% | 96 / 96 |
| SVM | 100% | 100% | 96 / 96 |
| Random Forest | 100% | 100% | 96 / 96 |

![Confusion matrices for KNN, SVM, and Random Forest](figures/yesno_confusion_matrices.png)

All three models correctly classified 43 No words and 53 Yes words. The results show that the MFCC features separated the two classes in this test split. The recordings came from one speaker, so the reported accuracy applies to held-out recordings from that speaker.

## 3. Voice Activity Detection on VOiCES

The VOiCES subset I downloaded contained **908 WAV files at 16 kHz**. I selected four recordings for the demonstration, one for each background condition. I used pretrained **Silero VAD** to estimate when speech was active in each recording.

| Background | Recording duration (s) | Predicted speech regions | Predicted speech duration (s) |
| :--- | ---: | ---: | ---: |
| No added distractor | 16.07 | 4 | 14.03 |
| Music | 16.16 | 6 | 13.48 |
| Television | 15.84 | 5 | 14.67 |
| Background babble | 15.84 | 5 | 14.80 |

![Waveforms showing the predicted speech regions in the four VOiCES recordings](figures/voices_vad.png)

The highlighted regions show when Silero predicted speech. For example, it marked six regions in the recording with music, covering 13.48 seconds. These durations summarize model predictions. Measuring VAD accuracy would require reference timestamps for speech.

## 4. Audio Event Detection on VOiCES

I applied pretrained **YAMNet** to the same four recordings. I kept the **Speech** and **Music** scores, used a threshold of **0.30**, and grouped consecutive windows that met the threshold into approximate event intervals.

![YAMNet speech and music scores over time for the four VOiCES recordings](figures/voices_events.png)

In the recording with music, YAMNet detected three music intervals.

| Music interval | Start (s) | End (s) | Peak score |
| :--- | ---: | ---: | ---: |
| 1 | 1.44 | 2.40 | 0.394 |
| 2 | 8.16 | 9.60 | 0.498 |
| 3 | 11.04 | 12.00 | 0.348 |

Speech and music can overlap because YAMNet scores the two classes separately. In the interactive demo, changing **Background** selects a different recording and updates the VAD summary, event timeline, and interval table.

## 5. Conclusion

The three YESNO classifiers gave the same result on the test set. Checking the segmentation was important because a merged or missing word can shift the labels for the following segments. On VOiCES, Silero estimated when speech was active, while YAMNet showed where it detected speech and music. The four examples illustrate the models' outputs under different recording conditions.

## 6. How to Run

1. Download [the notebook](Audio_Assignment.ipynb) and upload it to Google Colab.
2. Run the cells from top to bottom. The notebook downloads YESNO and the VOiCES subset.
3. Open the [interactive demo](https://markyou838-ctrl.github.io/digital-speech-processing/task8/index.html) to explore the saved results, or browse the [CSV files](results/) for the exported tables.
