"""
Anomaly detection for stock market data.
Implements three methods: Z-score, Isolation Forest, and Local Outlier Factor (LOF).
"""

import pandas as pd
import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies using multiple methods with agreement scoring."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.isolation_forest = None
        self.lof = None
        
        # Feature importance tracking
        self.feature_importance = {}
    
    def detect_anomalies(self, features_df: pd.DataFrame, 
                        feature_columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies using all three methods.
        
        Args:
            features_df: DataFrame with engineered features
            feature_columns: List of columns to use for detection (if None, use all numeric)
        
        Returns:
            DataFrame with anomaly flags and agreement scores
        """
        if features_df.empty:
            return pd.DataFrame()
        
        # Prepare features
        if feature_columns is None:
            # Use all numeric columns except date and target-like columns
            exclude_cols = ['date', 'symbol', 'anomaly', 'target']
            feature_columns = [col for col in features_df.columns 
                             if col not in exclude_cols and 
                             pd.api.types.is_numeric_dtype(features_df[col])]
        
        logger.info(f"Using {len(feature_columns)} features for anomaly detection")
        
        # Extract feature matrix
        X = features_df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction (optional)
        n_components = min(20, len(feature_columns))
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        logger.info(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # 1. Z-score method (univariate)
        z_score_flags = self._detect_z_score_anomalies(X, threshold=3.0)
        
        # 2. Isolation Forest (multivariate)
        isolation_flags = self._detect_isolation_forest_anomalies(X_pca)
        
        # 3. Local Outlier Factor (density-based)
        lof_flags = self._detect_lof_anomalies(X_pca)
        
        # Combine results
        result_df = features_df.copy()
        result_df['z_score_anomaly'] = z_score_flags
        result_df['isolation_forest_anomaly'] = isolation_flags
        result_df['lof_anomaly'] = lof_flags
        
        # Calculate agreement score (0-3)
        result_df['agreement_score'] = (
            result_df['z_score_anomaly'].astype(int) +
            result_df['isolation_forest_anomaly'].astype(int) +
            result_df['lof_anomaly'].astype(int)
        )
        
        # Calculate confidence score (weighted by agreement)
        result_df['confidence'] = result_df['agreement_score'] / 3.0
        
        # Add anomaly type based on which methods flagged it
        result_df['anomaly_type'] = result_df.apply(
            lambda row: self._get_anomaly_type(row), axis=1
        )
        
        # Calculate feature importance for anomalies
        self._calculate_feature_importance(X, feature_columns, result_df)
        
        logger.info(f"Detected {result_df['agreement_score'].sum()} total anomaly flags")
        logger.info(f"High-confidence anomalies (agreement >= 2): {(result_df['agreement_score'] >= 2).sum()}")
        
        return result_df
    
    def _detect_z_score_anomalies(self, X: pd.DataFrame, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies using Z-score method (univariate).
        
        Args:
            X: Feature matrix
            threshold: Z-score threshold for anomaly detection
        
        Returns:
            Boolean array of anomaly flags
        """
        z_scores = np.abs((X - X.mean()) / (X.std() + 1e-10))
        
        # Flag if any feature exceeds threshold
        anomaly_flags = (z_scores > threshold).any(axis=1).astype(int)
        
        logger.info(f"Z-score method flagged {anomaly_flags.sum()} anomalies")
        return anomaly_flags
    
    def _detect_isolation_forest_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            X: Feature matrix (scaled)
        
        Returns:
            Boolean array of anomaly flags (1 = anomaly, 0 = normal)
        """
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        # Isolation Forest returns -1 for anomalies, 1 for normal points
        predictions = self.isolation_forest.fit_predict(X)
        
        # Convert to 1 for anomalies, 0 for normal
        anomaly_flags = (predictions == -1).astype(int)
        
        logger.info(f"Isolation Forest flagged {anomaly_flags.sum()} anomalies")
        return anomaly_flags
    
    def _detect_lof_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor.
        
        Args:
            X: Feature matrix (scaled)
        
        Returns:
            Boolean array of anomaly flags
        """
        self.lof = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=False,  # We're using fit_predict
            n_neighbors=min(20, len(X) // 10)
        )
        
        # LOF returns -1 for anomalies, 1 for normal points
        predictions = self.lof.fit_predict(X)
        
        # Convert to 1 for anomalies, 0 for normal
        anomaly_flags = (predictions == -1).astype(int)
        
        logger.info(f"LOF flagged {anomaly_flags.sum()} anomalies")
        return anomaly_flags
    
    def _get_anomaly_type(self, row: pd.Series) -> str:
        """Determine anomaly type based on which methods flagged it."""
        flags = [
            ('z_score', row['z_score_anomaly']),
            ('isolation', row['isolation_forest_anomaly']),
            ('lof', row['lof_anomaly'])
        ]
        
        active_methods = [name for name, flag in flags if flag == 1]
        
        if not active_methods:
            return 'normal'
        
        if len(active_methods) == 3:
            return 'consensus_anomaly'
        elif 'z_score' in active_methods and len(active_methods) == 1:
            return 'statistical_outlier'
        elif 'isolation' in active_methods and 'lof' in active_methods:
            return 'density_anomaly'
        else:
            return 'partial_anomaly'
    
    def _calculate_feature_importance(self, X: pd.DataFrame, feature_names: List[str], 
                                    result_df: pd.DataFrame):
        """Calculate which features are most important for anomaly detection."""
        if self.isolation_forest is None:
            return
        
        try:
            # Get feature importance from Isolation Forest
            if hasattr(self.isolation_forest, 'feature_importances_'):
                importances = self.isolation_forest.feature_importances_
                
                # Map to original features (if using PCA, need to transform back)
                if self.pca is not None and len(importances) == self.pca.n_components_:
                    # Transform PCA importances back to original feature space
                    pca_components = self.pca.components_
                    original_importances = np.abs(pca_components).T @ importances
                    
                    # Normalize
                    original_importances = original_importances / original_importances.sum()
                    
                    # Store
                    for i, feature in enumerate(feature_names):
                        if i < len(original_importances):
                            self.feature_importance[feature] = original_importances[i]
                else:
                    # Direct feature importances
                    for i, feature in enumerate(feature_names):
                        if i < len(importances):
                            self.feature_importance[feature] = importances[i]
            
            # Also calculate mean absolute deviation for anomalies vs normal
            anomaly_mask = result_df['agreement_score'] >= 2
            if anomaly_mask.any():
                X_anomalies = X[anomaly_mask]
                X_normal = X[~anomaly_mask]
                
                if len(X_anomalies) > 1 and len(X_normal) > 1:
                    for i, feature in enumerate(feature_names):
                        if i < X.shape[1]:
                            mad = np.abs(X_anomalies.iloc[:, i].mean() - X_normal.iloc[:, i].mean())
                            self.feature_importance[feature] = self.feature_importance.get(feature, 0) + mad
            
            # Normalize importance scores
            if self.feature_importance:
                total = sum(self.feature_importance.values())
                if total > 0:
                    for feature in self.feature_importance:
                        self.feature_importance[feature] /= total
                
                # Sort by importance
                self.feature_importance = dict(
                    sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
                )
                
                logger.info("Top 5 important features for anomaly detection:")
                for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5]):
                    logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                    
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    def get_anomaly_summary(self, result_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for detected anomalies."""
        if result_df.empty:
            return {}
        
        summary = {
            'total_samples': len(result_df),
            'total_anomaly_flags': int(result_df[['z_score_anomaly', 'isolation_forest_anomaly', 'lof_anomaly']].sum().sum()),
            'z_score_anomalies': int(result_df['z_score_anomaly'].sum()),
            'isolation_forest_anomalies': int(result_df['isolation_forest_anomaly'].sum()),
            'lof_anomalies': int(result_df['lof_anomaly'].sum()),
            'high_confidence_anomalies': int((result_df['agreement_score'] >= 2).sum()),
            'consensus_anomalies': int((result_df['agreement_score'] == 3).sum()),
        }
        
        # Anomaly type distribution
        anomaly_types = result_df['anomaly_type'].value_counts().to_dict()
        summary['anomaly_type_distribution'] = anomaly_types
        
        # Date range of anomalies
        anomaly_dates = result_df[result_df['agreement_score'] >= 2]['date']
        if not anomaly_dates.empty:
            summary['first_anomaly_date'] = anomaly_dates.min().strftime('%Y-%m-%d')
            summary['last_anomaly_date'] = anomaly_dates.max().strftime('%Y-%m-%d')
            summary['anomaly_date_count'] = len(anomaly_dates.unique())
        
        # Feature importance summary
        if self.feature_importance:
            summary['top_features'] = list(self.feature_importance.keys())[:10]
        
        return summary
    
    def explain_anomaly(self, result_df: pd.DataFrame, index: int) -> Dict:
        """
        Provide explanation for a specific anomaly.
        
        Args:
            result_df: DataFrame with anomaly detection results
            index: Row index to explain
        
        Returns:
            Dictionary with explanation
        """
        if index >= len(result_df):
            return {"error": "Index out of bounds"}
        
        row = result_df.iloc[index]
        
        explanation = {
            'date': row['date'].strftime('%Y-%m-%d') if 'date' in row else str(index),
            'agreement_score': int(row['agreement_score']),
            'confidence': float(row['confidence']),
            'anomaly_type': row['anomaly_type'],
            'methods_flagged': [],
            'price_info': {},
            'feature_deviations': []
        }
        
        # Which methods flagged it
        if row['z_score_anomaly']:
            explanation['methods_flagged'].append('z_score')
        if row['isolation_forest_anomaly']:
            explanation['methods_flagged'].append('isolation_forest')
        if row['lof_anomaly']:
            explanation['methods_flagged'].append('lof')
        
        # Price information
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'daily_return']
        for col in price_cols:
            if col in row:
                explanation['price_info'][col] = float(row[col])
        
        return explanation

    # NEW METHODS ADDED FOR CS 210 PROJECT

    def classify_anomaly_type(self, anomaly_date, fomc_dates: List,
                             contagion_dates: Optional[List] = None,
                             news_df: Optional[pd.DataFrame] = None) -> str:
        """
        Classify anomaly type based on date, FOMC meetings, contagion dates, and news.

        Precedence: macroeconomic_event > sector_contagion > vader_sentiment_spike > unexplained.

        Args:
            anomaly_date: Date of the anomaly (datetime, date, or pandas Timestamp)
            fomc_dates: List of FOMC meeting dates (datetime.date)
            contagion_dates: List of dates flagged by contagion analysis (datetime.date)
            news_df: DataFrame with news data, must include 'date' and 'sentiment_compound'

        Returns:
            'macroeconomic_event' | 'sector_contagion' | 'vader_sentiment_spike' | 'unexplained'
        """
        try:
            ad = pd.to_datetime(anomaly_date).date()
        except Exception:
            return 'unexplained'

        # 1. Macroeconomic event (FOMC ±2 days)
        for fomc_date in fomc_dates:
            fd = fomc_date.date() if hasattr(fomc_date, 'date') else fomc_date
            if abs((ad - fd).days) <= 2:
                return 'macroeconomic_event'

        # 2. Sector contagion (date present in precomputed contagion list)
        if contagion_dates:
            for cd in contagion_dates:
                cd_norm = cd.date() if hasattr(cd, 'date') else cd
                if ad == cd_norm:
                    return 'sector_contagion'

        # 3. VADER sentiment spike (news volume + strong polarity within ±1 day)
        if news_df is not None and not news_df.empty and 'sentiment_compound' in news_df.columns:
            ad_ts = pd.Timestamp(ad)
            news_dates = pd.to_datetime(news_df['date'])
            mask = (news_dates >= ad_ts - pd.Timedelta(days=1)) & \
                   (news_dates <= ad_ts + pd.Timedelta(days=1))
            window_news = news_df[mask]
            if not window_news.empty:
                news_volume = len(window_news)
                avg_sentiment = window_news['sentiment_compound'].abs().mean()
                if news_volume >= 5 and avg_sentiment >= 0.3:
                    return 'vader_sentiment_spike'

        return 'unexplained'
    
    def precision_by_explanation(self, anomalies_df: pd.DataFrame) -> float:
        """
        Calculate the ratio of anomalies with non-unexplained classification.
        
        Args:
            anomalies_df: DataFrame with anomaly classifications
        
        Returns:
            Precision score (0.0 to 1.0)
        """
        if anomalies_df.empty:
            return 0.0
        
        # Count anomalies with non-unexplained classification
        explained_mask = anomalies_df['explained_type'] != 'unexplained'
        explained_count = explained_mask.sum()
        
        # Calculate precision
        precision = explained_count / len(anomalies_df)
        
        logger.info(f"Precision by explanation: {explained_count}/{len(anomalies_df)} = {precision:.3f}")
        return precision


if __name__ == "__main__":
    # Test anomaly detection
    import yfinance as yf
    from feature_engineering import FeatureEngineer
    
    print("Testing anomaly detection...")
    
    # Fetch and engineer sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo", interval="1d")
    df = df.reset_index()
    
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    
    print(f"Engineered features shape: {features_df.shape}")
    
    # Detect anomalies
    detector = AnomalyDetector(contamination=0.05)
    result_df = detector.detect_anomalies(features_df)
    
    print(f"\nAnomaly detection results:")
    print(f"Total samples: {len(result_df)}")
    print(f"Z-score anomalies: {result_df['z_score_anomaly'].sum()}")
    print(f"Isolation Forest anomalies: {result_df['isolation_forest_anomaly'].sum()}")
    print(f"LOF anomalies: {result_df['lof_anomaly'].sum()}")
    print(f"High-confidence anomalies (agreement >= 2): {(result_df['agreement_score'] >= 2).sum()}")
    
    # Show summary
    summary = detector.get_anomaly_summary(result_df)
    print(f"\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")