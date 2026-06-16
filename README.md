# StockAI-Sentiment
 
An AI-powered stock decision system that combines **5 machine learning models** with **real-time news sentiment analysis** (FinBERT) to predict Saudi Aramco (2222.SR) stock price movements and generate actionable trading recommendations.
 
---
 
## Overview
 
The system fetches historical stock data and live financial news, processes them through a dual pipeline (price prediction + sentiment analysis), and outputs a clear **BUY / SELL / HOLD** recommendation with a confidence score.
 
```
Historical Price Data (yfinance)
        +
Real-Time News (Google News RSS)
        |
        ▼
FinBERT Sentiment Analysis + Impact Score
        +
5 ML Models (LSTM · GRU · Transformer · XGBoost · Random Forest)
        |
        ▼
Weighted Ensemble → Decision Engine → BUY / SELL / HOLD + Confidence %
```
 
---
 
## Features
 
- **Multi-Model Forecasting** — 5 models trained in parallel: LSTM, GRU, Transformer, XGBoost, and Random Forest
- **Weighted Ensemble** — Final prediction is a weighted average of all models, weighted inversely by RMSE (best models contribute more)
- **FinBERT Sentiment** — Financial-domain NLP model analyzes Arabic and English news headlines in real-time
- **Impact Score** — Measures the actual market impact of each news article (price change × volume × sentiment direction alignment)
- **Time-Weighted Sentiment** — Recent news is weighted exponentially more than older articles (7-day rolling window)
- **Decision Engine** — Combines predicted price movement and combined signal (sentiment + impact) into a final trading signal
- **Confidence Scoring** — Dynamic confidence % based on model agreement, sentiment strength, and impact score
- **Visual Report** — Actual vs. predicted price charts for all models and the ensemble
---
 
## Tech Stack
 
| Component | Library |
|---|---|
| Stock Data | `yfinance` |
| News Scraping | `requests`, `beautifulsoup4`, `lxml` |
| Sentiment Analysis | `transformers` (ProsusAI/FinBERT) |
| Deep Learning | `tensorflow` / `keras` (LSTM, GRU, Transformer) |
| Machine Learning | `xgboost`, `scikit-learn` (XGBoost, Random Forest) |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib` |
 
---
 
## Model Architecture
 
### Deep Learning Models (sequence input)
 
```
Input: 60-day sequences × 7 features
       (Open, High, Low, Close, Volume, Sentiment, ImpactScore)
 
LSTM:
  LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(1)
 
GRU:
  GRU(128)  → Dropout(0.3) → GRU(64)  → Dropout(0.3) → Dense(1)
 
Transformer:
  MultiHeadAttention(heads=4, key_dim=32)
  → LayerNormalization
  → GlobalAveragePooling1D
  → Dense(64, relu) → Dropout(0.3) → Dense(1)
```
 
### Machine Learning Models (flattened input)
 
```
Input: flattened 60-day × 7 features → 420-dimensional vector
 
XGBoost:      n_estimators=300, max_depth=6, lr=0.05
RandomForest: n_estimators=200, max_depth=10
```
 
### Ensemble
 
All 5 models are combined into a **weighted ensemble** where each model's weight is inversely proportional to its RMSE on the test set, so the most accurate model contributes the most to the final prediction.
 
- **Optimizer:** Adam (deep learning models)
- **Loss:** Mean Squared Error (MSE)
- **Train/Test split:** 80% / 20%
- **Early Stopping:** patience=3 on validation loss
---
 
## Sentiment & Impact Pipeline
 
### News Sources
 
Google News RSS is queried with 12 search queries covering both English and Arabic terms related to Saudi Aramco, including earnings, dividends, oil production, and market activity.
 
### Sentiment Score (FinBERT)
 
Each news headline is passed through FinBERT and mapped to a score in [-1, +1]:
- Positive → +score
- Negative → -score
- Neutral  → 0
### Impact Score
 
For each news article, the actual market impact is calculated by looking at the price and volume the day after publication:
 
```
Impact = price_change_pct × 0.50
       + volume_ratio     × 0.30
       + direction_match  × 0.20
```
 
Where direction_match = +1 if the FinBERT sentiment direction matched the actual price movement, and -1 otherwise.
 
### Daily Aggregation
 
Sentiment and impact scores are aggregated per trading day using a **7-day rolling window** with exponential time-decay (more recent news carries higher weight).
 
---
 
## Decision Logic
 
The decision engine uses:
- `diff` = predicted price − last real price
- `vol` = 30-day price standard deviation
- `combined_signal` = sentiment × 0.4 + impact × 0.6
| Condition | Signal |
|---|---|
| diff > 0.5σ AND combined_signal > 0.2  | STRONG BUY  |
| diff > 0.3σ AND combined_signal > 0.15 | BUY         |
| diff < −0.5σ AND combined_signal < −0.2  | STRONG SELL |
| diff < −0.3σ AND combined_signal < −0.15 | SELL        |
| Otherwise | HOLD |
 
---
 
## Confidence Score
 
Confidence is calculated from three factors:
 
| Factor | Weight | Description |
|---|---|---|
| Model Agreement  | 50% | How closely the 5 models agree (based on coefficient of variation) |
| Sentiment Strength | 25% | Absolute value of the average sentiment score |
| Impact Strength  | 25% | Absolute value of the average impact score |
 
Final confidence is clamped between 1% and 99%.
 
---
 
## Output Example
 
```
AI STOCK REPORT — Saudi ARAMCO 2222
 
 REAL DATA
- Last Real Price  : 27.45 SAR
- Sentiment Avg    : 0.182
- Impact Score     : +0.091
- Combined Signal  : +0.127
 
 PREDICTION DATA
- Predicted Price  : 28.10 SAR
- Prediction Range : 27.63 — 28.57 SAR
- Model Std Dev    : ±0.237 SAR
- Model Agreement  : 88.2%
- Difference       : +0.65 SAR
- Confidence       : 74.3%
 
 FINAL RECOMMENDATION
 
   >>> BUY <<<
 
 ENSEMBLE WEIGHTS
  LSTM            22.4%  ██████████████████████
  GRU             19.8%  ███████████████████
  Transformer     18.1%  ██████████████████
  XGBoost         21.3%  █████████████████████
  RandomForest    18.4%  ██████████████████
```
 
---
 
## Installation
 
```bash
pip install yfinance beautifulsoup4 requests lxml transformers xgboost scikit-learn tensorflow matplotlib
```
 
---
 
## Usage
 
Open `StockAI-Sentiment.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially.
 
> **Note:** The first run downloads the FinBERT model (~500MB). An active internet connection is required for live news fetching and stock data retrieval.
 
---
 
## Evaluation Metrics
 
The notebook outputs **RMSE**, **MAE**, and **MAPE** on the test set for each individual model and the ensemble before generating the final recommendation.
 
---
 
## Disclaimer
 
This project is built for **educational and research purposes only**. It does not constitute financial advice. Always conduct your own research before making any investment decisions.
 
---
 
## Author
 
Built with Python, TensorFlow, XGBoost, and FinBERT.
