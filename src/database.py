"""
SQLite database schema for Anomalyy.
4-table design: stocks, price_data, news_articles, anomalies
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDatabase:
    """SQLite database manager for Anomalyy."""

    def __init__(self, db_path: str = "anomalyy.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    
    def _create_tables(self):
        """Create the 4 main tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Table 1: stocks - metadata for tracked stocks
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            industry TEXT,
            country TEXT,
            market_cap REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Table 2: price_data - historical price and volume data
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adj_close REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol) REFERENCES stocks(symbol),
            UNIQUE(symbol, date)
        )
        """)
        
        # Table 3: news_articles - news data with sentiment scores
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            published_at TIMESTAMP,
            title TEXT,
            description TEXT,
            source TEXT,
            url TEXT UNIQUE,
            sentiment_compound REAL,
            sentiment_positive REAL,
            sentiment_neutral REAL,
            sentiment_negative REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol) REFERENCES stocks(symbol)
        )
        """)
        
        # Table 4: anomalies - detected anomalies with agreement scores and explanation label
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            anomaly_date DATE,
            z_score_flag BOOLEAN,
            isolation_forest_flag BOOLEAN,
            lof_flag BOOLEAN,
            agreement_score INTEGER,
            confidence REAL,
            label TEXT,
            price_change_1d REAL,
            price_change_5d REAL,
            price_change_20d REAL,
            avg_sentiment REAL,
            news_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (symbol) REFERENCES stocks(symbol),
            UNIQUE(symbol, anomaly_date)
        )
        """)

        # Migrate existing DBs that predate the label column
        try:
            cursor.execute("ALTER TABLE anomalies ADD COLUMN label TEXT")
        except sqlite3.OperationalError:
            pass
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_data_symbol_date ON price_data(symbol, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_published ON news_articles(symbol, published_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_symbol_date ON anomalies(symbol, anomaly_date)")
        
        self.conn.commit()
        logger.info("Database tables created/verified successfully")
    
    def add_stock(self, symbol: str, name: str, sector: str = None, 
                  industry: str = None, country: str = "US", market_cap: float = None):
        """Add a stock to the tracking list."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO stocks (symbol, name, sector, industry, country, market_cap, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, name, sector, industry, country, market_cap, datetime.now()))
        self.conn.commit()
        logger.info(f"Added/updated stock: {symbol}")
    
    def add_price_data(self, symbol: str, df: pd.DataFrame):
        """Add price data for a stock."""
        # Ensure date column is datetime
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date'})
        
        df['symbol'] = symbol
        df['created_at'] = datetime.now()
        
        # Reorder columns to match table schema
        required_columns = ['symbol', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        if all(col in df.columns for col in required_columns):
            df_to_insert = df[required_columns].copy()
            df_to_insert.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            
            # Insert or replace
            df_to_insert.to_sql('price_data', self.conn, if_exists='append', index=False)
            logger.info(f"Added {len(df_to_insert)} price records for {symbol}")
        else:
            logger.warning(f"Missing required columns for {symbol}")
    
    def add_news_article(self, symbol: str, article_data: Dict):
        """Add a news article with sentiment scores."""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR IGNORE INTO news_articles 
        (symbol, published_at, title, description, source, url, 
         sentiment_compound, sentiment_positive, sentiment_neutral, sentiment_negative)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            article_data.get('published_at'),
            article_data.get('title'),
            article_data.get('description'),
            article_data.get('source'),
            article_data.get('url'),
            article_data.get('sentiment_compound'),
            article_data.get('sentiment_positive'),
            article_data.get('sentiment_neutral'),
            article_data.get('sentiment_negative')
        ))
        self.conn.commit()
    
    def add_anomaly(self, symbol: str, anomaly_date: str, z_score_flag: bool,
                   isolation_forest_flag: bool, lof_flag: bool, confidence: float,
                   price_changes: Dict = None, avg_sentiment: float = None,
                   news_count: int = 0, label: str = 'unexplained',
                   agreement_score: int = None, anomaly_type: str = None):
        """Add a detected anomaly. `anomaly_type` is accepted as an alias for `label`."""
        if anomaly_type is not None and label == 'unexplained':
            label = anomaly_type
        if agreement_score is None:
            agreement_score = sum([bool(z_score_flag), bool(isolation_forest_flag), bool(lof_flag)])
        if price_changes is None:
            price_changes = {}

        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO anomalies
        (symbol, anomaly_date, z_score_flag, isolation_forest_flag, lof_flag,
         agreement_score, confidence, label, price_change_1d, price_change_5d, price_change_20d,
         avg_sentiment, news_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            anomaly_date,
            int(bool(z_score_flag)),
            int(bool(isolation_forest_flag)),
            int(bool(lof_flag)),
            agreement_score,
            confidence,
            label,
            price_changes.get('1d', 0),
            price_changes.get('5d', 0),
            price_changes.get('20d', 0),
            avg_sentiment,
            news_count
        ))
        self.conn.commit()
        logger.info(f"Added anomaly for {symbol} on {anomaly_date} (agreement: {agreement_score}/3, label: {label})")
    
    def get_price_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve price data for a stock."""
        query = "SELECT * FROM price_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_anomalies(self, symbol: str = None, min_agreement: int = 2) -> pd.DataFrame:
        """Retrieve anomalies with optional filtering."""
        query = "SELECT * FROM anomalies WHERE agreement_score >= ?"
        params = [min_agreement]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY anomaly_date DESC"
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_news_for_period(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve news articles for a specific period."""
        query = """
        SELECT * FROM news_articles 
        WHERE symbol = ? AND published_at >= ? AND published_at <= ?
        ORDER BY published_at DESC
        """
        df = pd.read_sql_query(query, self.conn, params=(symbol, start_date, end_date))
        return df
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test the database creation
    with StockDatabase("test.db") as db:
        print("Database created successfully!")
        
        # Add a sample stock
        db.add_stock("AAPL", "Apple Inc.", "Technology", "Consumer Electronics", "US", 2.8e12)
        
        # Check tables
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables created: {[t[0] for t in tables]}")
DatabaseManager = StockDatabase  # alias for compatibility
