"""
Data ingestion pipeline for Anomalyy.
Uses yfinance for price data and GDELT 2.0 (via src.news_gdelt) for headlines.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
from .database import StockDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module constants as required
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'TSLA', 'AMZN', '^GSPC']
DEFAULT_START = '2015-01-01'
DEFAULT_END = '2024-12-31'


class DataIngestion:
    """Handles data collection from yfinance and NewsAPI."""
    
    def __init__(self, db_path: str = "anomalyy.db"):
        """Initialize with database connection."""
        self.db = StockDatabase(db_path)
    
    def fetch_stock_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data from yfinance.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with historical data
        """
        try:
            logger.info(f"Fetching data for {symbol} (period: {period}, interval: {interval})")
            
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Add metadata
            info = ticker.info
            stock_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            country = info.get('country', 'US')
            market_cap = info.get('marketCap')
            
            # Add stock to database
            self.db.add_stock(symbol, stock_name, sector, industry, country, market_cap)
            
            # Add price data to database
            self.db.add_price_data(symbol, df)
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_stock_data_by_date(self, symbol: str, start_date: str = DEFAULT_START,
                                end_date: str = DEFAULT_END, interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data from yfinance using date range.

        Returns a DataFrame with lowercase columns
        ('date','open','high','low','close','adj_close','volume') so downstream
        consumers (FeatureEngineer) can use it directly. The DB write happens
        before normalization since `add_price_data` expects yfinance's
        title-case columns.
        """
        try:
            logger.info(f"Fetching data for {symbol} ({start_date} to {end_date}, interval: {interval})")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = df.reset_index()

            # Newer yfinance versions auto-adjust Close and drop "Adj Close".
            # Synthesize it so add_price_data's required-columns check passes.
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']

            info = ticker.info
            stock_name = info.get('longName', symbol)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            country = info.get('country', 'US')
            market_cap = info.get('marketCap')

            self.db.add_stock(symbol, stock_name, sector, industry, country, market_cap)
            self.db.add_price_data(symbol, df)

            # Normalize to lowercase column names for downstream FeatureEngineer
            column_map = {'Date': 'date', 'Open': 'open', 'High': 'high',
                          'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                          'Adj Close': 'adj_close'}
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

            logger.info(f"Successfully fetched {len(df)} records for {symbol} from {start_date} to {end_date}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "2y", 
                             interval: str = "1d", delay: float = 1.0):
        """
        Fetch data for multiple stocks with rate limiting.
        
        Args:
            symbols: List of stock ticker symbols
            period: Time period
            interval: Data interval
            delay: Delay between requests (seconds)
        """
        results = {}
        
        for symbol in symbols:
            df = self.fetch_stock_data(symbol, period, interval)
            results[symbol] = df
            
            # Rate limiting
            time.sleep(delay)
        
        return results
    
    def fetch_default_stocks(self, start_date: str = DEFAULT_START, 
                            end_date: str = DEFAULT_END, interval: str = "1d",
                            delay: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all default tickers using date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            delay: Delay between requests (seconds)
        
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        logger.info(f"Fetching default stocks ({start_date} to {end_date})")
        return self.fetch_multiple_stocks_by_date(DEFAULT_TICKERS, start_date, end_date, interval, delay)
    
    def fetch_multiple_stocks_by_date(self, symbols: List[str], start_date: str, 
                                     end_date: str, interval: str = "1d", 
                                     delay: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks using date range with rate limiting.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            delay: Delay between requests (seconds)
        """
        results = {}
        
        for symbol in symbols:
            df = self.fetch_stock_data_by_date(symbol, start_date, end_date, interval)
            results[symbol] = df
            
            # Rate limiting
            time.sleep(delay)
        
        return results
    
    def fetch_news_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch news articles for a stock from GDELT 2.0 across [start_date, end_date].

        Returns NewsAPI-shape dicts; pass directly into `_process_and_store_news`.
        """
        from .news_gdelt import fetch_for_ticker
        return fetch_for_ticker(symbol, start_date, end_date)

    def update_all_data(self, symbols: List[str], update_frequency: str = "daily"):
        """
        Update all data for tracked stocks based on frequency.

        Args:
            symbols: List of stock symbols to update
            update_frequency: 'daily', 'weekly', or 'monthly'
        """
        logger.info(f"Starting {update_frequency} data update for {len(symbols)} stocks")

        if update_frequency == "daily":
            period = "1mo"
        elif update_frequency == "weekly":
            period = "3mo"
        else:
            period = "1y"

        price_results = self.fetch_multiple_stocks(symbols, period=period, delay=1.5)
        logger.info(f"Data update completed. Price data for {len(price_results)} stocks.")
        return price_results
    
    def _process_and_store_news(self, symbol: str, articles: List[Dict]):
        """Process news articles with VADER sentiment analysis and store in database."""
        if not articles:
            return
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            
            for article in articles:
                # Extract relevant fields
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title}. {description}"
                
                # Get sentiment scores
                sentiment = analyzer.polarity_scores(content)
                
                # Prepare article data
                article_data = {
                    'published_at': article.get('publishedAt'),
                    'title': title,
                    'description': description,
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url'),
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_positive': sentiment['pos'],
                    'sentiment_neutral': sentiment['neu'],
                    'sentiment_negative': sentiment['neg']
                }
                
                # Store in database
                self.db.add_news_article(symbol, article_data)
                
        except ImportError:
            logger.warning("vaderSentiment not installed. Skipping sentiment analysis.")
        except Exception as e:
            logger.error(f"Error processing news for {symbol}: {e}")
    
    def close(self):
        """Close database connection."""
        self.db.close()


if __name__ == "__main__":
    # Test the data ingestion
    ingestor = DataIngestion("test_ingestion.db")
    
    # Test with a few stocks
    test_symbols = ["AAPL", "MSFT", "GOOGL"]
    
    print("Testing data ingestion...")
    results = ingestor.fetch_multiple_stocks(test_symbols, period="1mo", delay=2)
    
    for symbol, df in results.items():
        if not df.empty:
            print(f"{symbol}: {len(df)} records, from {df['Date'].min()} to {df['Date'].max()}")
    
    ingestor.close()
    print("Test completed!")