# Cryptocurrency Tweet Topic Classification

A comprehensive machine learning project demonstrating natural language processing and deep learning techniques for automated topic classification of cryptocurrency-related tweets.

## Project Overview

This project showcases end-to-end data science methodology, from data collection through model deployment, focusing on cryptocurrency discourse analysis on Twitter. The work demonstrates proficiency in data preprocessing, unsupervised learning (topic modeling), supervised learning (neural networks), and model evaluation.

### Key Accomplishments

- **Processed 2GB+ of raw social media data** into a clean, structured dataset
- **Achieved 98.19% classification accuracy** using custom neural network architecture
- **Discovered meaningful topic patterns** in cryptocurrency discourse through LDA modeling
- **Implemented robust data preprocessing pipeline** handling text normalization, spell correction, and feature engineering
- **Designed scalable ML workflow** suitable for production deployment

### Technical Scope

This project covers:
- **Data Engineering**: Large-scale text data processing and cleaning
- **Natural Language Processing**: Text preprocessing, topic modeling with LDA
- **Deep Learning**: Neural network design and optimization with PyTorch
- **Model Evaluation**: Performance analysis and validation

## 📁 Project Structure

```
├── data/                           # Data files
│   ├── raw/                        # Original, unprocessed data
│   │   └── crypto_tweets_october_2020.json
│   ├── processed/                  # Cleaned and processed datasets
│   │   ├── crypto_tweets_final.csv
│   │   ├── crypto_tweets_preprocessed.csv
│   │   ├── crypto_tweets_spell_checked.csv
│   │   └── crypto_tweets_stratified_sample.csv
│   ├── crypto_dataset_description.rtf
│   └── crypto_keywords.rtf
├── models/                         # Trained models
│   └── crypto_tweet_classifier.pth
├── notebooks/                      # Jupyter notebooks
│   ├── crypto_tweet_sentiment_analysis.ipynb  # Main analysis
│   ├── text_preprocessing.ipynb               # Text processing utilities
│   └── exploratory_analysis.ipynb             # Initial data exploration
├── src/                           # Source code (future development)
├── docs/                          # Documentation and assets
│   └── dataset_screenshot.png
└── README.md                      # This file
```

## 🔍 Dataset Description

**Source**: Twitter API v1.1 real-time stream
**Time Period**: October 10, 2020 - March 3, 2021
**Collection Method**: Stream-watcher using cryptocurrency keywords
**Storage**: AWS MongoDB server
**Format**: JSON objects (MongoDB export)

### Keywords & Cryptocurrencies

The dataset focuses on tweets containing relevant cryptocurrency keywords, filtered for coins that remained significant through 2020:

- **Bitcoin** (`Bitcoin`, `$BTC`)
- **Ethereum** (`Ethereum`, `$ETH`)  
- **Binance Coin** (`$BNB`)
- **ChainLink** (`ChainLink`, `$LINK`)
- **Litecoin** (`Litecoin`, `$LTC`)
- **Cardano** (`$ADA`)
- **And many others** (see `data/crypto_keywords.rtf`)

## 🧹 Data Processing Pipeline

### 1. Keyword Extraction & Filtering
- Extracted unique cryptocurrency keywords from tweets
- Filtered for relevant coins still active in 2020
- Created balanced dataset through stratified sampling

### 2. Stratified Sampling
- Target sample size: 270,000 tweets
- Maintained balanced representation across cryptocurrencies
- Final dataset: 253,049 tweets

### 3. Text Preprocessing
- **Language Detection**: Filtered for English tweets
- **Spell Checking**: Applied TextBlob correction
- **Text Cleaning**:
  - Removed URLs, @mentions, emojis
  - Preserved cashtags (e.g., `$BTC`)
  - Tokenization and lemmatization
  - Stopword removal
  - HTML entity replacement

## 🧠 Methodology

### Topic Modeling (LDA)
- **Algorithm**: Latent Dirichlet Allocation
- **Topics**: 4 main themes discovered
- **Parameters**: 25 passes, vocabulary filtering
- **Output**: Topic distributions and top words per topic

#### Discovered Topics:
1. **Topic 0**: Cryptocurrency symbols and market terms (`$ETH`, `$BTC`, `price`, `market`)
2. **Topic 1**: Blockchain technology and transactions (`cardano`, `ethereum`, `blockchain`, `transaction`)
3. **Topic 2**: DeFi and token finance (`token`, `finance`, `protocol`, `staking`)
4. **Topic 3**: Social engagement and giveaways (`join`, `follow`, `giveaway`, `community`)

### Neural Network Classification

**Architecture**:
- Input Layer: 27,519 neurons (vocabulary size)
- Hidden Layer 1: 128 neurons + ReLU + Dropout (50%)
- Hidden Layer 2: 64 neurons + ReLU + Dropout (50%)
- Output Layer: 4 neurons (topics)

**Training Configuration**:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr=0.0001, weight_decay=1e-5)
- Batch Size: 32
- Epochs: 20
- Data Split: 70% train, 15% validation, 15% test

## 📈 Results

### Model Performance
- **Test Accuracy**: 98.19%
- **Test Precision**: 98.17%
- **Test Recall**: 97.98%

The model demonstrates excellent performance in classifying crypto tweets into the discovered topic categories, with balanced precision and recall metrics.

### Topic Analysis
The LDA model successfully identified distinct themes in crypto Twitter discourse:
- Market-focused discussions (prices, trading)
- Technical blockchain conversations
- DeFi and financial protocols
- Community engagement and promotions

## 🛠 Technologies Used

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **NLP**: nltk, textblob, langdetect, gensim
- **Machine Learning**: scikit-learn, torch (PyTorch)
- **Visualization**: matplotlib, seaborn, wordcloud
- **Data Storage**: JSON, CSV

## 📋 Requirements

```python
# Core libraries
pandas
numpy
matplotlib
seaborn

# NLP & Text Processing
nltk
textblob
langdetect
gensim
wordcloud

# Machine Learning
scikit-learn
torch
torchvision

# Additional utilities
json
re
random
```

## Key Insights & Impact

### Technical Achievements
- **Scalable Data Processing**: Successfully handled and processed 2GB+ of unstructured social media data
- **High-Performance Classification**: Achieved 98%+ accuracy through careful architecture design and hyperparameter optimization
- **Meaningful Pattern Discovery**: LDA modeling revealed distinct, interpretable themes in cryptocurrency discourse
- **Production-Ready Pipeline**: Developed robust preprocessing and modeling workflow suitable for real-world deployment

### Business Applications
This work demonstrates capabilities directly applicable to:
- **Social Media Analytics**: Brand monitoring and sentiment tracking
- **Financial Technology**: Market sentiment analysis for trading algorithms
- **Content Moderation**: Automated classification of social media content
- **Customer Intelligence**: Understanding community conversations and trends

### Research Contributions
- Demonstrated effectiveness of combining unsupervised (LDA) and supervised (neural networks) learning for social media analysis
- Validated the importance of comprehensive text preprocessing in achieving high model performance
- Provided insights into cryptocurrency community discourse patterns and themes

---

*This project demonstrates proficiency in end-to-end machine learning development, from data engineering through model deployment, with a focus on natural language processing and deep learning applications.*