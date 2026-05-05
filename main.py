"""
main.py - End-to-end pipeline for Anomalyy
CS 210: Data Management for Data Science
Author: Purabh Singh
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Windows consoles default to cp1252 and crash on '✓' (U+2713).
# Force UTF-8 on stdout/stderr so logging and progress prints work everywhere.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline():
    from src.data_ingestion import DataIngestion, DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END
    from src.feature_engineering import FeatureEngineer
    from src.anomaly_detection import AnomalyDetector
    from src.database import StockDatabase
    from src.visualization import Visualization
    from src.fomc_events import get_fomc_dates, label_fomc_anomalies
    from src.contagion_analysis import cross_stock_correlation, detect_sector_contagion

    print("\n" + "="*60)
    print("  ANOMALYY PIPELINE")
    print("  CS 210: Data Management for Data Science")
    print("="*60 + "\n")

    db_path = Path(__file__).parent / "anomalyy.db"
    viz_dir = Path(__file__).parent / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    db = StockDatabase(str(db_path))
    ingestion = DataIngestion()
    engineer = FeatureEngineer()
    detector = AnomalyDetector()
    visualizer = Visualization(str(viz_dir))
    fomc_dates = get_fomc_dates()

    price_data_dict = {}
    features_dict = {}
    all_results = {}
    anomalies_dict = {}

    # Step 1: Fetch data
    print(f"[1/6] Fetching stock data ({DEFAULT_START} to {DEFAULT_END})...")
    for ticker in DEFAULT_TICKERS:
        try:
            df = ingestion.fetch_stock_data_by_date(ticker, start_date=DEFAULT_START, end_date=DEFAULT_END)
            if df is None or df.empty:
                logger.warning(f"No data for {ticker}")
                continue
            price_data_dict[ticker] = df
            db.add_stock(ticker, ticker)
            db.add_price_data(ticker, df)
            print(f"  ✓ {ticker}: {len(df)} trading days")
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")

    if not price_data_dict:
        print("ERROR: No stock data retrieved.")
        return

    # Step 2: Feature engineering
    print(f"\n[2/6] Engineering features...")
    for ticker, df in price_data_dict.items():
        try:
            features = engineer.engineer_features(df)
            features_dict[ticker] = features
            print(f"  ✓ {ticker}: {features.shape[1]} features")
        except Exception as e:
            logger.error(f"Feature engineering failed for {ticker}: {e}")

    # Step 3: Anomaly detection
    print(f"\n[3/6] Running anomaly detection (Z-Score + Isolation Forest + LOF)...")
    for ticker, features in features_dict.items():
        try:
            results = detector.detect_anomalies(features)
            all_results[ticker] = results
            n = int(results['agreement_score'].ge(2).sum()) if 'agreement_score' in results.columns else 0
            print(f"  ✓ {ticker}: {n} high-confidence anomalies")
        except Exception as e:
            logger.error(f"Anomaly detection failed for {ticker}: {e}")

    # Step 4: Build anomalies dict for contagion + classify types
    print(f"\n[4/6] Classifying anomaly types...")
    for ticker, results in all_results.items():
        if 'agreement_score' in results.columns:
            high_conf = results[results['agreement_score'] >= 2].copy()
        else:
            high_conf = results.copy()
        if not high_conf.empty:
            idx = high_conf.index if high_conf.index.name == 'date' else high_conf.index
            high_conf_reset = high_conf.reset_index() if high_conf.index.name else high_conf
            anomalies_dict[ticker] = high_conf_reset

    contagion_dates = []
    try:
        contagion_dates = detect_sector_contagion(anomalies_dict)
        print(f"  ✓ Sector contagion dates found: {len(contagion_dates)}")
    except Exception as e:
        logger.error(f"Contagion analysis failed: {e}")

    news_dict = {}
    print(f"\n[4a/6] Fetching GDELT news ({DEFAULT_START} to {DEFAULT_END})...")
    for ticker in all_results.keys():
        try:
            articles = ingestion.fetch_news_data(ticker, start_date=DEFAULT_START, end_date=DEFAULT_END)
            ingestion._process_and_store_news(ticker, articles)
            news_df = db.get_news_for_period(ticker, DEFAULT_START, DEFAULT_END)
            if not news_df.empty:
                news_df = news_df.rename(columns={'published_at': 'date'})
                news_dict[ticker] = news_df
            print(f"  ✓ {ticker}: {len(articles)} articles fetched, {len(news_df)} stored")
        except Exception as e:
            logger.error(f"News fetch failed for {ticker}: {e}")

    for ticker, results in all_results.items():
        dates = results['date'] if 'date' in results.columns else pd.to_datetime(results.index)
        ticker_news = news_dict.get(ticker)
        types = [
            detector.classify_anomaly_type(d, fomc_dates,
                                           contagion_dates=contagion_dates,
                                           news_df=ticker_news)
            for d in dates
        ]
        all_results[ticker]['anomaly_type'] = types

    # Step 5: Save to database
    print(f"\n[5/6] Saving to database...")
    for ticker, results in all_results.items():
        try:
            high_conf = results[results['agreement_score'] >= 2] \
                if 'agreement_score' in results.columns else results
            for _, row in high_conf.iterrows():
                try:
                    anomaly_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    db.add_anomaly(
                        symbol=ticker,
                        anomaly_date=anomaly_date,
                        z_score_flag=bool(row.get('z_score_anomaly', False)),
                        isolation_forest_flag=bool(row.get('isolation_forest_anomaly', False)),
                        lof_flag=bool(row.get('lof_anomaly', False)),
                        agreement_score=int(row.get('agreement_score', 0)),
                        confidence=float(row.get('confidence', 0)),
                        label=row.get('anomaly_type', 'unexplained'),
                    )
                except Exception as e:
                    logger.debug(f"Row save failed for {ticker}: {e}")
            print(f"  ✓ {ticker}: {len(high_conf)} anomalies saved")
        except Exception as e:
            logger.error(f"DB save failed for {ticker}: {e}")

    # Step 6: Visualizations
    print(f"\n[6/6] Generating visualizations...")
    for ticker, results in all_results.items():
        try:
            price_df = features_dict.get(ticker, price_data_dict[ticker])
            visualizer.plot_price_with_anomalies(price_df, results, ticker)
            print(f"  ✓ {ticker} chart saved")
        except Exception as e:
            logger.error(f"Visualization failed for {ticker}: {e}")

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY REPORT")
    print("="*60)
    total = 0
    type_counts = {'macroeconomic_event': 0, 'vader_sentiment_spike': 0,
                   'sector_contagion': 0, 'unexplained': 0}
    for ticker, results in all_results.items():
        hc = results[results['agreement_score'] >= 2] if 'agreement_score' in results.columns else results
        n = len(hc)
        total += n
        print(f"\n  {ticker}: {n} anomalies")
        if 'anomaly_type' in hc.columns:
            for t, c in hc['anomaly_type'].value_counts().items():
                type_counts[t] = type_counts.get(t, 0) + c
                print(f"    - {t}: {c}")

    explained = sum(v for k, v in type_counts.items() if k != 'unexplained')
    precision = explained / total if total > 0 else 0
    print(f"\n  TOTAL ANOMALIES: {total}")
    print(f"  PRECISION-BY-EXPLANATION: {precision:.1%} ({explained}/{total} explained)")
    print(f"  DATABASE: {db_path}")
    print(f"  CHARTS: {viz_dir}/")
    print("\n" + "="*60 + "\n")
    db.close()


if __name__ == '__main__':
    run_pipeline()
