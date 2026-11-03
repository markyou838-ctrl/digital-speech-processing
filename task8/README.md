# Task 8 · Digital Speech Processing

- [Colab notebook](Audio_Assignment.ipynb)
- [Interactive results source](index.html)
- [CSV results](results/)

Open the notebook in Colab and run the cells from top to bottom. The notebook downloads YESNO and the VOiCES subset. For the interactive view, open `index.html` locally in a browser or serve the folder through GitHub Pages.

## YESNO classification

KNN, SVM and Random Forest each classified 96 of 96 held-out words correctly. The split held out 12 recordings from the same speaker. Features are the mean and standard deviation of 13 MFCCs.

![Confusion matrices](figures/yesno_confusion_matrices.png)

## VOiCES

Silero VAD estimates speech activity. YAMNet detects speech and music using a score threshold of 0.30. Both models were applied to four example recordings, one per background. The exported intervals are model predictions.

![Speech activity](figures/voices_vad.png)

![Event scores](figures/voices_events.png)
