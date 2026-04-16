# StockAI-Aramco 📈

An AI-powered stock decision system that combines **LSTM deep learning** with **real-time news sentiment analysis** (FinBERT) to predict Saudi Aramco (2222.SR) stock price movements and generate actionable trading recommendations.

---

## Overview

This system fetches historical stock data and live financial news, processes them through a dual-pipeline (price prediction + sentiment analysis), and outputs a clear **BUY / SELL / HOLD** recommendation with a confidence score.

```
Historical Price Data (yfinance)
        +
Real-Time News (Google News RSS)
        |
        ▼
FinBERT Sentiment Analysis
        +
LSTM Neural Network (60-day sequences)
        |
        ▼
Decision Engine → BUY / SELL / HOLD + Confidence %
```

---

## Features

- **LSTM Price Forecasting** — 2-layer LSTM network trained on historical OHLCV data
- **FinBERT Sentiment** — Financial-domain NLP model analyzes news headlines in real-time
- **Time-Weighted Sentiment** — Recent news weighted exponentially more than older articles
- **Decision Engine** — Combines predicted price movement + sentiment score into a final signal
- **Confidence Scoring** — Dynamic confidence % based on price delta, sentiment strength, and model accuracy
- **Visual Report** — Actual vs. predicted price chart with full evaluation metrics

---

## Tech Stack

| Component | Library |
|---|---|
| Stock Data | `yfinance` |
| News Scraping | `requests`, `beautifulsoup4`, `lxml` |
| Sentiment Analysis | `transformers` (ProsusAI/FinBERT) |
| Deep Learning | `tensorflow` / `keras` (LSTM) |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib` |

---

## Model Architecture

```
Input: 60-day sequences × 6 features
       (Open, High, Low, Close, Volume, Sentiment)

LSTM(128) → Dropout(0.3)
    ↓
LSTM(64)  → Dropout(0.3)
    ↓
Dense(1)  → Predicted Close Price
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Train/Test split:** 80% / 20%

---

## Decision Logic

| Condition | Signal |
|---|---|
| Predicted ↑ > 0.5σ AND sentiment > 0.2 | STRONG BUY |
| Predicted ↑ > 0.3σ | BUY |
| Predicted ↓ < −0.5σ AND sentiment < −0.2 | STRONG SELL |
| Predicted ↓ < −0.3σ | SELL |
| Otherwise | HOLD |

---

## Output Example

```
AI STOCK REPORT — Saudi ARAMCO 2222

 REAL DATA
- Last Real Price : 27.45 SAR
- Sentiment Avg   : 0.182

 PREDICTION DATA
- Predicted Price : 28.10 SAR
- Difference      : +0.65 SAR
- Confidence      : 74.3%

 FINAL RECOMMENDATION
   >>> BUY <<<
```

---

## Installation

```bash
pip install yfinance beautifulsoup4 requests lxml transformers tensorflow scikit-learn matplotlib
```

---

## Usage

Open `AISDS_StockAI.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially.

> **Note:** First run downloads the FinBERT model (~500MB). An internet connection is required for live news fetching.

---

## Evaluation Metrics

The notebook outputs **RMSE**, **MAE**, and **MAPE** on the test set to measure prediction accuracy before generating the final recommendation.

---

## Disclaimer

This project is built for **educational and research purposes only**. It does not constitute financial advice. Always do your own research before making investment decisions.

---

## Author

Built with Python, TensorFlow, and FinBERT.
