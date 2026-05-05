"""
Feature engineering for Anomalyy.
Calculates technical indicators and derived features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from raw price data for anomaly detection."""
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: List of window sizes for moving averages and other indicators
        """
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50, 200]  # Common technical analysis windows
        
        self.window_sizes = window_sizes
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from raw price data.
        
        Args:
            df: DataFrame with columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        
        Returns:
            DataFrame with original data plus engineered features
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Ensure date is datetime and sort
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df = result_df.sort_values('date').reset_index(drop=True)
        
        # 1. Basic price transformations
        result_df = self._add_basic_features(result_df)
        
        # 2. Technical indicators
        result_df = self._add_technical_indicators(result_df)
        
        # 3. Statistical features
        result_df = self._add_statistical_features(result_df)
        
        # 4. Volume-based features
        result_df = self._add_volume_features(result_df)
        
        # 5. Price pattern features
        result_df = self._add_pattern_features(result_df)
        
        # Remove rows with NaN values (from rolling windows)
        initial_len = len(result_df)
        result_df = result_df.dropna()
        logger.info(f"Removed {initial_len - len(result_df)} rows with NaN values")
        
        return result_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price transformations."""
        result = df.copy()
        
        # Daily returns
        result['daily_return'] = result['close'].pct_change()
        result['log_return'] = np.log(result['close'] / result['close'].shift(1))
        
        # Price ratios
        result['high_low_ratio'] = result['high'] / result['low']
        result['close_open_ratio'] = result['close'] / result['open']
        
        # Normalized price (0-1 range over lookback window)
        lookback = 252  # Trading year
        result['price_normalized'] = (result['close'] - result['close'].rolling(lookback).min()) / \
                                    (result['close'].rolling(lookback).max() - result['close'].rolling(lookback).min() + 1e-10)
        
        return result
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        result = df.copy()
        
        # Simple Moving Averages
        for window in self.window_sizes:
            result[f'sma_{window}'] = result['close'].rolling(window=window).mean()
            result[f'ema_{window}'] = result['close'].ewm(span=window, adjust=False).mean()
            
            # Price relative to moving averages
            result[f'price_vs_sma_{window}'] = result['close'] / result[f'sma_{window}'] - 1
            result[f'price_vs_ema_{window}'] = result['close'] / result[f'ema_{window}'] - 1
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = result['close'].rolling(window=window).mean()
            std = result['close'].rolling(window=window).std()
            
            result[f'bb_upper_{window}'] = sma + (2 * std)
            result[f'bb_lower_{window}'] = sma - (2 * std)
            result[f'bb_width_{window}'] = result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']
            result[f'bb_position_{window}'] = (result['close'] - result[f'bb_lower_{window}']) / \
                                            (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}'] + 1e-10)
        
        # Average True Range (ATR) - volatility measure
        result['tr'] = np.maximum(
            result['high'] - result['low'],
            np.maximum(
                abs(result['high'] - result['close'].shift(1)),
                abs(result['low'] - result['close'].shift(1))
            )
        )
        result['atr_14'] = result['tr'].rolling(window=14).mean()
        
        # Relative Strength Index (RSI)
        for window in [14, 28]:
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            result[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = result['close'].ewm(span=12, adjust=False).mean()
        ema_26 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        return result
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        result = df.copy()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            result[f'volatility_{window}'] = result['log_return'].rolling(window=window).std() * np.sqrt(252)
            result[f'skew_{window}'] = result['log_return'].rolling(window=window).skew()
            result[f'kurtosis_{window}'] = result['log_return'].rolling(window=window).kurt()
            
            # Z-score of returns
            result[f'return_zscore_{window}'] = (result['log_return'] - result['log_return'].rolling(window=window).mean()) / \
                                               (result['log_return'].rolling(window=window).std() + 1e-10)
            
            # Maximum drawdown in window
            rolling_max = result['close'].rolling(window=window, min_periods=1).max()
            result[f'drawdown_{window}'] = (result['close'] - rolling_max) / rolling_max
        
        # Autocorrelation features
        for lag in [1, 2, 5]:
            result[f'autocorr_{lag}'] = result['log_return'].autocorr(lag=lag)
        
        # Hurst exponent (roughness measure)
        result['hurst_20'] = self._calculate_hurst(result['close'], window=20)
        
        return result
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        result = df.copy()
        
        # Volume indicators
        result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
        result['volume_ratio'] = result['volume'] / (result['volume_sma_20'] + 1e-10)
        
        # Price-Volume correlation
        for window in [5, 10, 20]:
            result[f'price_volume_corr_{window}'] = result['close'].rolling(window=window).corr(result['volume'])
        
        # On-Balance Volume (OBV)
        result['obv'] = 0
        mask = result['close'] > result['close'].shift(1)
        result.loc[mask, 'obv'] = result['volume']
        result.loc[~mask, 'obv'] = -result['volume']
        result['obv'] = result['obv'].cumsum()
        
        # Volume-weighted average price
        result['vwap'] = (result['volume'] * (result['high'] + result['low'] + result['close']) / 3).cumsum() / \
                        result['volume'].cumsum()
        
        return result
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern recognition features."""
        result = df.copy()
        
        # Gap features
        result['gap_up'] = (result['open'] > result['close'].shift(1)).astype(int)
        result['gap_down'] = (result['open'] < result['close'].shift(1)).astype(int)
        result['gap_size'] = (result['open'] / result['close'].shift(1)) - 1
        
        # Doji pattern (small body relative to range)
        body_size = abs(result['close'] - result['open'])
        total_range = result['high'] - result['low']
        result['doji'] = (body_size / (total_range + 1e-10) < 0.1).astype(int)
        
        # Marubozu pattern (large body relative to range)
        result['marubozu'] = (body_size / (total_range + 1e-10) > 0.9).astype(int)
        
        # Engulfing patterns
        prev_body = abs(result['close'].shift(1) - result['open'].shift(1))
        result['bullish_engulfing'] = ((result['close'] > result['open']) & 
                                      (result['close'].shift(1) < result['open'].shift(1)) &
                                      (result['open'] < result['close'].shift(1)) &
                                      (result['close'] > result['open'].shift(1))).astype(int)
        
        result['bearish_engulfing'] = ((result['close'] < result['open']) & 
                                      (result['close'].shift(1) > result['open'].shift(1)) &
                                      (result['open'] > result['close'].shift(1)) &
                                      (result['close'] < result['open'].shift(1))).astype(int)
        
        return result
    
    def _calculate_hurst(self, series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Hurst exponent for roughness estimation."""
        hurst_values = []
        
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
            
            window_data = series.iloc[i-window:i].values
            if len(window_data) < window:
                hurst_values.append(np.nan)
                continue
            
            # Simplified Hurst calculation
            lags = range(2, min(20, len(window_data)))
            tau = []
            
            for lag in lags:
                # Variance of differences
                diff = np.diff(window_data, lag)
                if len(diff) > 1:
                    tau.append(np.std(diff))
                else:
                    tau.append(np.nan)
            
            tau = np.array(tau)
            lags = np.array(lags)
            
            # Remove NaN values
            mask = ~np.isnan(tau)
            if np.sum(mask) > 2:
                # Fit to power law
                try:
                    hurst = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
                    hurst_values.append(hurst)
                except:
                    hurst_values.append(np.nan)
            else:
                hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=series.index)
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features organized by category."""
        categories = {
            'price_basic': ['daily_return', 'log_return', 'high_low_ratio', 'close_open_ratio', 'price_normalized'],
            'moving_averages': [f'sma_{w}' for w in self.window_sizes] + [f'ema_{w}' for w in self.window_sizes],
            'bollinger_bands': ['bb_upper_20', 'bb_lower_20', 'bb_width_20', 'bb_position_20',
                               'bb_upper_50', 'bb_lower_50', 'bb_width_50', 'bb_position_50'],
            'volatility': ['atr_14', 'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50'],
            'momentum': ['rsi_14', 'rsi_28', 'macd', 'macd_signal', 'macd_histogram'],
            'statistical': ['skew_5', 'skew_10', 'skew_20', 'skew_50',
                           'kurtosis_5', 'kurtosis_10', 'kurtosis_20', 'kurtosis_50',
                           'return_zscore_5', 'return_zscore_10', 'return_zscore_20', 'return_zscore_50',
                           'drawdown_5', 'drawdown_10', 'drawdown_20', 'drawdown_50',
                           'autocorr_1', 'autocorr_2', 'autocorr_5', 'hurst_20'],
            'volume': ['volume_ratio', 'price_volume_corr_5', 'price_volume_corr_10', 'price_volume_corr_20',
                      'obv', 'vwap'],
            'patterns': ['gap_up', 'gap_down', 'gap_size', 'doji', 'marubozu', 
                        'bullish_engulfing', 'bearish_engulfing']
        }
        
        return categories


if __name__ == "__main__":
    # Test feature engineering
    import yfinance as yf
    
    print("Testing feature engineering...")
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo", interval="1d")
    df = df.reset_index()
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    
    print(f"\nEngineered data shape: {features_df.shape}")
    print(f"Number of features: {len(features_df.columns)}")
    
    # Show feature categories
    categories = engineer.get_feature_categories()
    for category, features in categories.items():
        print(f"\n{category.upper()} ({len(features)} features):")
        print(f"  {features[:5]}{'...' if len(features) > 5 else ''}")
    
    print("\nFeature engineering test completed!")