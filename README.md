# sentiment-analysis
# Twitter Sentiment Analysis Project

## Overview

This project implements a deep learning-based sentiment analysis model using the Sentiment140 dataset to classify tweets as positive or negative.

## Dataset

- **Source**: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/data)
- **Description**: Contains 1.6 million tweets with sentiment labels
- **Labels**: 
  - 0: Negative sentiment
  - 4: Positive sentiment

## Project Structure

### Dependencies
- pandas
- numpy
- re
- nltk
- matplotlib
- tensorflow
- scikit-learn

### Key Processing Steps
1. Data Cleaning
   - Remove HTML tags
   - Remove special characters
   - Normalize text
2. Text Tokenization
3. Stopwords Removal
4. Word Embedding (Using GloVe 50d)
5. Sequence Padding

## Model Architecture

### Neural Network Details
- **Type**: LSTM with Attention Mechanism
- **Layers**:
  - Embedding Layer
  - LSTM Layer
  - Attention Mechanism
  - Dropout Layer
  - Dense Layers
- **Output**: Binary Sentiment Classification

### Model Performance
- Training Accuracy: Up to 85.56%
- Validation Accuracy: Around 81-82%

## How to Use

### Preprocessing Pipeline
```python
def pipline(text):
    # Cleans and preprocesses input text
    # Converts text to model input
    # Returns sentiment prediction
    pass
```

### Example Usage
```python
# Positive sentiment examples
result = pipline("Absolutely love it! Worth every penny.")
result = pipline("This product exceeded my expectations!")

# Negative sentiment examples
result = pipline("This product is a complete waste of money.")
result = pipline("The item arrived late and was defective.")
```

## Key Components

- **Data Preprocessing**: Tokenization, normalization
- **Feature Extraction**: Word embeddings using GloVe
- **Model**: LSTM with custom attention mechanism
- **Evaluation**: Accuracy and loss tracking

## Limitations
- Works best with English language tweets
- Sentiment classification is binary (positive/negative)
- Performance may vary with different types of text

## Future Improvements
- Experiment with more advanced embedding techniques
- Try transformer-based models
- Implement multi-class sentiment classification
- Enhance preprocessing for better generalization

## License
Please refer to the original dataset's licensing terms.

## Acknowledgements
- Dataset: Sentiment140
- Embedding: GloVe (Stanford NLP)
