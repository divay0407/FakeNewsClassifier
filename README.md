# SMS Spam Detection with LSTM

**Notebook:** `FakenewsLSTM.ipynb`
**Stack:** Python, TensorFlow/Keras, NLTK, scikit-learn

> **Note on the filename:** this notebook is named `FakenewsLSTM.ipynb`, but the code it contains trains a **binary spam/ham (SMS spam) classifier**, not a fake-news detector. It reads `mail_data.csv` with `Category` (`ham`/`spam`) and `Message` columns. You may want to rename the file to something like `SpamDetectionLSTM.ipynb` to match what it actually does, or swap in a fake-news dataset if that was the original intent.

## Overview

This notebook builds a deep learning pipeline that classifies SMS text messages as **spam** or **ham** (not spam) using an **LSTM (Long Short-Term Memory)** neural network. It covers the full workflow: text cleaning, word embedding, model training, and evaluation, plus a helper function to classify new, arbitrary messages.

## Dataset

- **File:** `mail_data.csv` (expected at `/content/mail_data.csv`, i.e. Google Colab path)
- **Columns:**
  - `Category` — label, `ham` or `spam`
  - `Message` — raw SMS text
- **Size:** 5,572 messages, no missing values

## Pipeline

1. **Load & inspect data** — `pandas` read, null check, feature/label split (`x` = Message, `y` = Category).
2. **Text preprocessing** (per message):
   - Strip non-alphabetic characters with regex
   - Lowercase
   - Tokenize into words
   - Remove English stopwords (`nltk.corpus.stopwords`)
   - Apply Porter stemming (`nltk.stem.porter.PorterStemmer`)
   - Rejoin into a cleaned string (the resulting cleaned corpus)
3. **Vectorize text:**
   - One-hot encode each message into integer word indices with a vocabulary size of **5,000** (`tensorflow.keras.preprocessing.text.one_hot`)
   - Pad/truncate all sequences to a fixed length of **20** tokens (`pad_sequences`, post-padding)
4. **Encode labels:** one-hot via `pd.get_dummies(y, drop_first=True)` → binary target (1 = spam, 0 = ham).
5. **Train/test split:** 67% / 33% (`test_size=0.33`, `random_state=42`).
6. **Model architecture** (Keras `Sequential`):

   | Layer | Config |
   |---|---|
   | Embedding | `input_dim=5000`, `output_dim=40`, `input_length=20` |
   | Dropout | 0.3 |
   | LSTM | 100 units |
   | Dropout | 0.3 |
   | Dense | 1 unit, sigmoid activation |

   - **Loss:** `binary_crossentropy`
   - **Optimizer:** `adam`
   - **Metric:** `accuracy`
   - **Total params:** 769,505 (256,501 trainable)

7. **Training:** 10 epochs, batch size 64, with validation on the held-out test set.
8. **Evaluation:** confusion matrix, accuracy score, and full classification report via `scikit-learn`.
9. **Inference helper:** `predict_message(message, model)` — cleans and vectorizes a new raw text string the same way as training data and returns `"Spam"` or `"Ham"` (threshold: probability > 0.6 → spam).

## Results

On the held-out test set (1,839 messages):

| Metric | Value |
|---|---|
| **Accuracy** | **97.7%** |
| Precision (spam) | 0.95 |
| Recall (spam) | 0.88 |
| F1-score (spam) | 0.91 |

```
Confusion Matrix
[[1581   12]
 [  30  216]]
```

Training accuracy climbed to ~99.9% by epoch 9–10, while validation accuracy plateaued around 97–98%, suggesting mild overfitting in later epochs — worth watching if extending training time.

## Requirements

```
pandas
numpy
tensorflow
nltk
scikit-learn
```

On first run, the notebook downloads the NLTK `stopwords` corpus:
```python
import nltk
nltk.download("stopwords")
```

## Usage

1. Place `mail_data.csv` at `/content/mail_data.csv` (or update the path if running outside Google Colab).
2. Run all cells top to bottom.
3. Use the final cell's `predict_message()` function to classify your own text:
   ```python
   predict_message("Congratulations! You've won a free prize, click here", model)
   # -> "Spam"
   ```

## Possible Improvements

- Rename the notebook/repo to reflect its actual task (spam detection), or substitute a genuine fake-news dataset if that was the goal.
- Increase `sent_length` for longer messages, and consider pretrained embeddings (e.g., GloVe) instead of random one-hot + trainable embedding.
- Add early stopping / learning-rate scheduling to curb the overfitting seen after epoch ~5.
- Use `Tokenizer` + `texts_to_sequences` instead of `one_hot` to avoid hash collisions in the vocabulary.
- Save the trained model (`model.save(...)`) and tokenizer/vocab for reuse without retraining.
