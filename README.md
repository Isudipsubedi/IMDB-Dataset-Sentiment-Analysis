# Sentiment Analysis with LSTM As a part of NLP Challenge by Fellowship.ai

## Overview
This project implements a Sentiment Analysis model using a Long Short-Term Memory (LSTM) network. The dataset used is the IMDB Movie Reviews dataset, which contains 50,000 reviews labeled as positive or negative.

## Key Features
- Preprocessing of text data including tokenization and padding.
- Built an LSTM-based deep learning model for binary sentiment classification.
- Visualized performance metrics such as accuracy, loss, confusion matrix, and ROC curve.
- Predicted sentiments of new user-input reviews.

---

## Dataset
The dataset contains:
- **50,000 reviews** (25,000 positive, 25,000 negative).
- File: `IMDB Dataset.csv`

### Data Columns
- **review**: Text of the movie review.
- **sentiment**: Sentiment label (Positive/Negative).

---

## Requirements
Install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn tensorflow wordcloud scikit-learn
```

---

## Project Structure
```
📂 Sentiment Analysis Project
├── 📄 IMDB Dataset.csv     # Dataset
├── 📄 sentiment_analysis.py # Main implementation
├── 📄 README.md            # Project documentation
```

---

## Implementation Steps

1. **Data Loading and Exploration**
   - Load the dataset.
   - Visualize class distribution.

2. **Text Preprocessing**
   - Tokenized reviews using Keras' `Tokenizer`.
   - Padded sequences to a fixed length (200).

3. **Model Building**
   - Embedding Layer for feature extraction.
   - LSTM Layer for sequential data processing.
   - Dense Layer with sigmoid activation for classification.

4. **Model Training**
   - Compiled with Adam optimizer and binary cross-entropy loss.
   - Trained for 5 epochs with validation.

5. **Evaluation and Analysis**
   - Tested model on a separate test set.
   - Generated confusion matrix and classification report.
   - Plotted ROC curve.

6. **Visualization**
   - Accuracy and loss trends.
   - Word clouds for positive and negative reviews.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/username/sentiment-analysis-lstm.git
   cd sentiment-analysis-lstm
   ```

2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

3. Test the model with custom input reviews:
   ```python
   predict_sentiment("The movie was fantastic and full of emotions!")
   ```

---

## Results

### Metrics
- **Accuracy**: 87.4%
- **Loss**: ~0.31

### Visualizations
- **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
- **ROC Curve**: Demonstrates the model's performance at various thresholds.
- **Word Clouds**: Displays frequent words in positive and negative reviews.

---

## Model Architecture

```text
Embedding Layer:  Input (5000) -> Output (128)
LSTM Layer:       Units (128) with Dropout (0.2)
Dense Layer:      Units (1) with Sigmoid Activation
```

---

## Future Work
- Enhance preprocessing by removing stopwords and stemming.
- Experiment with hyperparameter tuning (e.g., learning rate, LSTM units).
- Compare LSTM with other models (e.g., GRU, BERT).

---

## Acknowledgments
- Dataset: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Libraries: TensorFlow, Keras, Scikit-learn, WordCloud

---

## Author
**Sudip Subedi**  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourlinkedin)

