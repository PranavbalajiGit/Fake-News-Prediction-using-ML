# Fake News Prediction using Machine Learning

A machine learning project that classifies news articles as real or fake using Natural Language Processing (NLP) and Logistic Regression.

## Overview

This project implements a binary classification model to detect fake news articles. It uses TF-IDF vectorization for feature extraction and Logistic Regression for classification, achieving perfect accuracy on the provided dataset.

## Dataset Description

The dataset contains the following columns:

- **id**: Unique identifier for each news article
- **title**: The headline of the news article
- **author**: Author name of the article
- **text**: The full text content of the article (may be incomplete)
- **label**: Binary classification label
  - `0`: Real news
  - `1`: Fake news

## Technical Approach

### 1. Data Preprocessing

**Feature Engineering**: A new feature called `content` is created by combining the author name and title. This combined feature captures both the source credibility and headline characteristics, which are strong indicators of fake news.

```python
content = author + ' ' + title
```

### 2. Text Processing Pipeline

**Stemming Process**: The text undergoes several transformation steps:

- **Special Character Removal**: All non-alphabetic characters are removed using regex pattern `[^a-zA-Z]`
- **Lowercase Conversion**: Text is converted to lowercase for consistency
- **Tokenization**: Text is split into individual words
- **Stopword Removal**: Common English words (like "the", "is", "and") are removed as they don't contribute to meaning
- **Stemming**: Words are reduced to their root form using Porter Stemmer
  - Example: "running" → "run", "discoveries" → "discoveri"

### 3. Feature Extraction

**TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency):
- Converts text data into numerical vectors
- Assigns weights to words based on their importance
- Higher weight for words that are frequent in a document but rare across all documents
- Produces sparse matrix representation of the text data

### 4. Model Training

**Logistic Regression Classifier**:
- Binary classification algorithm suitable for this two-class problem
- Trained on 80% of the data (8,000 samples)
- Uses the TF-IDF vectors as input features

### 5. Model Evaluation

The dataset is split using stratified sampling (80-20 split) to maintain the same proportion of real and fake news in both training and testing sets.

**Results**:
- **Training Accuracy**: 100%
- **Testing Accuracy**: 100%

## Dependencies

```python
numpy
pandas
scikit-learn
nltk
re (standard library)
```

## Installation

1. Install required packages:
```bash
pip install numpy pandas scikit-learn nltk
```

2. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage

1. Prepare your dataset in CSV format with columns: `id`, `title`, `author`, `text`, `label`
2. Update the file path in the code:
```python
news_dataset = pd.read_csv('Fake News Dataset.csv')
```
3. Run the notebook cells sequentially

## Key Functions

### `stemming(content)`
Processes text through the complete NLP pipeline:
- Removes special characters
- Converts to lowercase
- Removes stopwords
- Applies Porter Stemming
- Returns cleaned and stemmed text

## Model Performance Notes

The perfect accuracy (100% on both training and testing) suggests:
- The dataset may be relatively simple or small (10,000 samples)
- Clear distinguishing patterns between real and fake news
- Possible data leakage or overfitting concerns in real-world applications

**Recommendation**: For production use, validate on a larger, more diverse dataset and consider using cross-validation for more robust evaluation.

## Limitations

- No handling of missing values (dataset is complete)
- Binary classification only (real vs fake)
- May not generalize well to unseen news domains
- Perfect accuracy indicates possible data simplicity

## Future Improvements

- Add cross-validation for better generalization assessment
- Implement additional models (Random Forest, SVM, Deep Learning)
- Include text content in features, not just author and title
- Add confusion matrix and precision/recall metrics
- Handle missing values with imputation techniques
- Test on more diverse, real-world datasets

## License

This project is open source and available under the [MIT License](LICENSE)

## Author

**PRANAV BALAJI P MA**
- GitHub: [@PranavbalajiGit](https://github.com/PranavbalajiGit)
