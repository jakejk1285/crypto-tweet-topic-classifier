# Crypto Tweet Sentiment Analysis

A machine learning project that analyzes crypto-related tweets and classifies them into topic categories using Latent Dirichlet Allocation (LDA) and neural networks.

## 📊 Project Overview

This project explores the relationship between cryptocurrency sentiment on Twitter and market behavior by:

1. **Collecting and filtering crypto-related tweets** from October 2020
2. **Preprocessing and cleaning** tweet text data
3. **Applying topic modeling (LDA)** to discover latent themes
4. **Training a neural network classifier** to predict tweet topics
5. **Evaluating model performance** on unseen data

## 🎯 Research Objectives

- Analyze crypto sentiment patterns in social media data
- Develop automated topic classification for cryptocurrency tweets
- Explore potential relationships between social sentiment and price movements
- Create a reproducible pipeline for crypto sentiment analysis

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

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Download NLTK data**: 
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt')
   ```
4. **Run notebooks**: Start with `notebooks/crypto_tweet_sentiment_analysis.ipynb`

## 🔬 Future Work

- Expand dataset to include more recent crypto trends
- Integrate real-time price data for sentiment-price correlation analysis
- Implement more sophisticated deep learning architectures
- Deploy model as a web service for live tweet classification
- Add support for additional cryptocurrencies and languages

## 📊 Key Insights

1. **Social Media Patterns**: Crypto Twitter shows distinct conversation patterns around trading, technology, and community
2. **Classification Success**: High-accuracy automated topic classification is achievable for crypto tweets
3. **Preprocessing Impact**: Careful text preprocessing is crucial for model performance
4. **Topic Coherence**: LDA successfully captures meaningful thematic differences in crypto discourse

## 📝 Citation

If you use this project in your research, please cite:

```
Crypto Tweet Sentiment Analysis
Author: [Your Name]
Year: 2020-2021
GitHub: [Repository URL]
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

*This project was completed as part of an undergraduate research initiative exploring the intersection of social media sentiment and cryptocurrency markets.*