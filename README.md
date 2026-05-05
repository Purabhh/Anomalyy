# Anomalyy
### CS 210: Data Management for Data Science

A data pipeline that detects anomalous trading days in historical stock data using three complementary unsupervised machine learning methods, then explains detected anomalies using FOMC meeting dates, news sentiment, and cross-stock contagion analysis.

---

## Installation

```bash
git clone https://github.com/Purabhh/Anomalyy.git
cd Anomalyy
pip install -r requirements.txt
```

No API keys required. News headlines come from [GDELT 2.0](https://www.gdeltproject.org/), a free open public-news index.

---

## Quick Start

```bash
python main.py
```

This runs the full pipeline: fetches 10 years of data for AAPL, MSFT, TSLA, AMZN, and ^GSPC, detects anomalies, classifies them, and outputs a summary report + visualizations.

---

## Project Structure

```
Anomalyy/
├── main.py                     # End-to-end pipeline runner
├── requirements.txt
├── README.md
├── src/
│   ├── data_ingestion.py       # yfinance price data fetching
│   ├── news_gdelt.py           # GDELT 2.0 DOC API news fetcher
│   ├── feature_engineering.py  # 50+ technical indicators
│   ├── anomaly_detection.py    # Z-Score, Isolation Forest, LOF
│   ├── database.py             # SQLite schema + CRUD
│   ├── visualization.py        # Plotly interactive charts
│   ├── fomc_events.py          # FOMC meeting dates 2015-2024
│   └── contagion_analysis.py   # Cross-stock correlation + contagion
├── notebooks/
│   └── analysis.ipynb          # Exploratory analysis notebook
├── tests/
│   └── test_pipeline.py        # pytest test suite
└── visualizations/             # Generated charts (created on run)
```

---

## Methodology

### Data Sources
- **Historical prices**: `yfinance` — daily OHLCV data for AAPL, MSFT, TSLA, AMZN, ^GSPC (Jan 2015 – Dec 2024)
- **News headlines**: GDELT 2.0 DOC API — global financial news indexed since Feb 2015, scored with VADER for sentiment
- **FOMC dates**: Hardcoded list of all ~80 Fed meeting dates from 2015–2024

### Feature Engineering (50+ features)
From raw OHLCV data, the pipeline computes:
- **Returns**: daily log returns, rolling returns
- **Moving averages**: SMA/EMA for 5, 10, 20, 50, 200-day windows
- **Volatility**: ATR-14, rolling annualized volatility
- **Bollinger Bands**: upper/lower bands, bandwidth, position
- **Momentum**: RSI-14/28, MACD, MACD histogram
- **Volume**: OBV, VWAP, volume ratio
- **Statistical**: skewness, kurtosis, z-scores, Hurst exponent

### Anomaly Detection (Three Methods)
Three methods are applied and their results combined via **agreement scoring** (0–3):

| Method | Type | Key Parameter |
|--------|------|--------------|
| Z-Score Thresholding | Statistical | threshold = 3.0 σ |
| Isolation Forest | Ensemble/Tree | contamination = 10% |
| Local Outlier Factor | Density-based | n_neighbors = 20 |

Anomalies confirmed by **≥ 2 methods** are classified as high-confidence. This multi-method approach reduces false positives and improves robustness.

### Anomaly Classification
Each detected anomaly is assigned one of four types:
- **macroeconomic_event** — date within 2 days of an FOMC meeting
- **vader_sentiment_spike** — high news volume + strong VADER sentiment (|compound| > 0.3)
- **sector_contagion** — 3+ stocks flagged within a 3-day window
- **unexplained** — none of the above conditions met

### Evaluation
Since this is unsupervised, conventional metrics don't apply. Instead, we use:
- **Precision-by-explanation**: % of anomalies that match a known market event (FOMC, earnings, contagion)
- **Cross-method agreement rate**: % of anomalies flagged by ≥ 2 methods

---

## Database Schema

Four normalized SQLite tables:

```sql
stocks(symbol PK, name, sector, industry, country, market_cap, created_at, last_updated)

price_data(id PK, symbol FK, date, open, high, low, close, adj_close, volume, created_at)
  UNIQUE(symbol, date)

news_articles(id PK, symbol FK, published_at, title, description, source, url UNIQUE,
              sentiment_compound, sentiment_positive, sentiment_neutral, sentiment_negative)

anomalies(id PK, symbol FK, anomaly_date, z_score_flag, isolation_forest_flag, lof_flag,
          agreement_score, confidence, anomaly_type, fomc_related,
          price_change_1d, price_change_5d, avg_sentiment, news_count)
  UNIQUE(symbol, anomaly_date)
```

---

## Limitations

1. **GDELT coverage start date**: The GDELT 2.0 DOC index begins Feb 18, 2015. The first ~6 weeks of the analysis window (Jan 1 – Feb 17, 2015) have no news coverage; the `vader_sentiment_spike` classifier cannot fire there.
2. **GDELT keyword precision**: Headlines are matched by company name + ticker against a whitelist of major financial-news domains. Off-topic hits are reduced but not eliminated.
3. **yfinance data quality**: Adjusted close prices may differ from raw prices. Split/dividend adjustments applied automatically.
4. **Unsupervised evaluation**: No ground-truth anomaly labels exist, so precision-by-explanation is a proxy metric, not a true precision score.
5. **Contamination assumption**: Isolation Forest assumes 10% contamination — this hyperparameter significantly affects sensitivity.
6. **Survivorship bias**: Only currently-listed tickers are analyzed.

---

## Citations

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys, 41*(3), 1–58.

Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review. *Applied Soft Computing, 90*, 106181.

Nassirtoussi, A. K., Aghabozorgi, S., Wah, T. Y., & Ngo, D. C. L. (2014). Text mining for market prediction: A systematic review. *Expert Systems with Applications, 41*(16), 7653–7670.
