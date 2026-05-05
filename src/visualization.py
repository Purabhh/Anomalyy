"""
Visualization module for Anomalyy.
Generates interactive charts and dashboards using Plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualization:
    """Handles all visualization tasks for the anomaly detection project."""
    
    def __init__(self, output_dir: str = "visualizations"):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_price_with_anomalies(self, features_df: pd.DataFrame, 
                                 anomalies_df: pd.DataFrame, 
                                 symbol: str = "Stock",
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive price chart with anomalies highlighted.
        
        Args:
            features_df: DataFrame with price data and features
            anomalies_df: DataFrame with anomaly detection results
            symbol: Stock symbol for title
            save_path: Optional path to save the figure
        
        Returns:
            Plotly figure object
        """
        if features_df.empty:
            logger.warning("Empty features dataframe provided")
            return go.Figure()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f"{symbol} Price with Anomalies", "Volume"),
            shared_xaxes=True
        )
        
        # Add price trace (candlestick or line)
        if all(col in features_df.columns for col in ['open', 'high', 'low', 'close']):
            # Use candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=features_df['date'],
                    open=features_df['open'],
                    high=features_df['high'],
                    low=features_df['low'],
                    close=features_df['close'],
                    name="Price",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        else:
            # Use line chart
            fig.add_trace(
                go.Scatter(
                    x=features_df['date'],
                    y=features_df['close'] if 'close' in features_df.columns else features_df.iloc[:, 0],
                    mode='lines',
                    name='Price',
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
        
        # Highlight anomalies if available
        if not anomalies_df.empty and 'agreement_score' in anomalies_df.columns:
            high_conf_anomalies = anomalies_df[anomalies_df['agreement_score'] >= 2]
            
            if not high_conf_anomalies.empty:
                # Add anomaly markers
                fig.add_trace(
                    go.Scatter(
                        x=high_conf_anomalies['date'],
                        y=high_conf_anomalies['close'] if 'close' in high_conf_anomalies.columns else high_conf_anomalies.iloc[:, 0],
                        mode='markers',
                        name='High-Confidence Anomalies',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='diamond',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate='<b>Anomaly Detected</b><br>' +
                                     'Date: %{x}<br>' +
                                     'Price: %{y:.2f}<br>' +
                                     'Agreement Score: %{text}<extra></extra>',
                        text=high_conf_anomalies['agreement_score']
                    ),
                    row=1, col=1
                )
        
        # Add volume trace
        if 'volume' in features_df.columns:
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(features_df['close'], features_df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=features_df['date'],
                    y=features_df['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Price Chart with Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2_title="Volume",
            hovermode='x unified',
            template='plotly_dark',
            height=800,
            showlegend=True
        )
        
        # Update x-axis ranges
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Price chart saved to: {save_path}")
        
        return fig
    
    def plot_feature_correlation(self, features_df: pd.DataFrame, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation heatmap of engineered features.
        
        Args:
            features_df: DataFrame with engineered features
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure object
        """
        if features_df.empty:
            logger.warning("Empty features dataframe provided")
            return plt.figure()
        
        # Select numeric columns for correlation
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation heatmap")
            return plt.figure()
        
        # Calculate correlation matrix
        corr_matrix = features_df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                title: str = "Correlation Heatmap",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap for cross-stock correlation matrix.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Plot title
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure object
        """
        if corr_matrix.empty:
            logger.warning("Empty correlation matrix provided")
            return plt.figure()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8}
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_anomaly_distribution(self, anomalies_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Create bar chart showing anomaly distribution by type.
        
        Args:
            anomalies_df: DataFrame with anomaly detection results
            save_path: Optional path to save the figure
        
        Returns:
            Plotly figure object
        """
        if anomalies_df.empty or 'anomaly_type' not in anomalies_df.columns:
            logger.warning("No anomaly type data available")
            return go.Figure()
        
        # Count anomalies by type
        type_counts = anomalies_df['anomaly_type'].value_counts().reset_index()
        type_counts.columns = ['anomaly_type', 'count']
        
        # Create bar chart
        fig = px.bar(
            type_counts,
            x='anomaly_type',
            y='count',
            color='anomaly_type',
            title='Anomaly Distribution by Type',
            labels={'anomaly_type': 'Anomaly Type', 'count': 'Count'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            template='plotly_dark',
            xaxis_title="Anomaly Type",
            yaxis_title="Count",
            showlegend=False,
            height=500
        )
        
        # Add value labels on bars
        fig.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Anomaly distribution chart saved to: {save_path}")
        
        return fig
    
    def plot_method_comparison(self, anomalies_df: pd.DataFrame,
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Create visualization comparing the three anomaly detection methods.
        
        Args:
            anomalies_df: DataFrame with anomaly detection results
            save_path: Optional path to save the figure
        
        Returns:
            Plotly figure object
        """
        if anomalies_df.empty:
            logger.warning("Empty anomalies dataframe provided")
            return go.Figure()
        
        # Prepare data for stacked bar chart
        method_cols = ['z_score_anomaly', 'isolation_forest_anomaly', 'lof_anomaly']
        
        # Check which method columns exist
        available_cols = [col for col in method_cols if col in anomalies_df.columns]
        
        if not available_cols:
            logger.warning("No method anomaly columns found")
            return go.Figure()
        
        # Count anomalies by method
        method_counts = {}
        for col in available_cols:
            method_name = col.replace('_anomaly', '').replace('_', ' ').title()
            method_counts[method_name] = int(anomalies_df[col].sum())
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(method_counts.keys()),
                y=list(method_counts.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                text=list(method_counts.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Anomaly Detection Method Comparison',
            xaxis_title="Detection Method",
            yaxis_title="Number of Anomalies Detected",
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Method comparison chart saved to: {save_path}")
        
        return fig
    
    def plot_contagion_events(self, contagion_events: Dict[datetime, List[str]],
                             anomalies_data: Dict[str, pd.DataFrame],
                             save_path: Optional[str] = None) -> go.Figure:
        """
        Create timeline visualization of contagion events.
        
        Args:
            contagion_events: Dictionary of contagion events
            anomalies_data: Dictionary of anomaly data by ticker
            save_path: Optional path to save the figure
        
        Returns:
            Plotly figure object
        """
        if not contagion_events:
            logger.warning("No contagion events provided")
            return go.Figure()
        
        # Prepare data for timeline
        timeline_data = []
        
        for event_date, tickers in contagion_events.items():
            # Count total anomalies across affected stocks on event date
            total_anomalies = 0
            for ticker in tickers:
                if ticker in anomalies_data and not anomalies_data[ticker].empty:
                    # Count anomalies within ±1 day of event date
                    mask = (anomalies_data[ticker]['date'] >= event_date - timedelta(days=1)) & \
                           (anomalies_data[ticker]['date'] <= event_date + timedelta(days=1))
                    if 'agreement_score' in anomalies_data[ticker].columns:
                        high_conf = anomalies_data[ticker].loc[mask, 'agreement_score'] >= 2
                        total_anomalies += high_conf.sum()
            
            timeline_data.append({
                'date': event_date,
                'num_stocks': len(tickers),
                'tickers': ', '.join(tickers),
                'total_anomalies': total_anomalies
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        if timeline_df.empty:
            return go.Figure()
        
        # Create timeline visualization
        fig = go.Figure()
        
        # Add scatter points (size based on number of stocks)
        fig.add_trace(go.Scatter(
            x=timeline_df['date'],
            y=timeline_df['num_stocks'],
            mode='markers+text',
            marker=dict(
                size=timeline_df['num_stocks'] * 5 + 10,
                color=timeline_df['total_anomalies'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Total Anomalies")
            ),
            text=timeline_df['tickers'],
            textposition="top center",
            hovertemplate='<b>Contagion Event</b><br>' +
                         'Date: %{x}<br>' +
                         'Stocks Affected: %{y}<br>' +
                         'Total Anomalies: %{marker.color}<br>' +
                         'Tickers: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sector Contagion Events Timeline',
            xaxis_title="Date",
            yaxis_title="Number of Stocks Affected",
            template='plotly_dark',
            height=600,
            hovermode='closest'
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Contagion events timeline saved to: {save_path}")
        
        return fig
    
    def plot_fomc_alignment(self, anomalies_df: pd.DataFrame,
                           fomc_dates: List[datetime],
                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create visualization showing alignment of anomalies with FOMC meetings.
        
        Args:
            anomalies_df: DataFrame with anomaly detection results
            fomc_dates: List of FOMC meeting dates
            save_path: Optional path to save the figure
        
        Returns:
            Plotly figure object
        """
        if anomalies_df.empty:
            logger.warning("Empty anomalies dataframe provided")
            return go.Figure()
        
        # Filter for high-confidence anomalies
        high_conf_anomalies = anomalies_df[anomalies_df['agreement_score'] >= 2].copy()
        
        if high_conf_anomalies.empty:
            logger.warning("No high-confidence anomalies found")
            return go.Figure()
        
        # Label FOMC-related anomalies
        from fomc_events import label_fomc_anomalies
        labeled_df = label_fomc_anomalies(high_conf_anomalies)
        
        # Prepare data for visualization
        fomc_data = []
        for fomc_date in fomc_dates:
            # Count anomalies within ±2 days of FOMC date
            mask = (labeled_df['date'] >= fomc_date - timedelta(days=2)) & \
                   (labeled_df['date'] <= fomc_date + timedelta(days=2))
            fomc_anomalies = labeled_df[mask]
            
            fomc_data.append({
                'fomc_date': fomc_date,
                'total_anomalies': len(fomc_anomalies),
                'fomc_related': fomc_anomalies['fomc_related'].sum() if 'fomc_related' in fomc_anomalies.columns else 0
            })
        
        fomc_df = pd.DataFrame(fomc_data)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Anomalies Around FOMC Meetings", "FOMC-Related Anomalies"),
            vertical_spacing=0.15
        )
        
        # Plot 1: Total anomalies around FOMC dates
        fig.add_trace(
            go.Bar(
                x=fomc_df['fomc_date'],
                y=fomc_df['total_anomalies'],
                name='Total Anomalies',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Plot 2: FOMC-related anomalies
        fig.add_trace(
            go.Bar(
                x=fomc_df['fomc_date'],
                y=fomc_df['fomc_related'],
                name='FOMC-Related',
                marker_color='red'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Anomaly Alignment with FOMC Meetings',
            template='plotly_dark',
            height=700,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="FOMC Meeting Date", row=2, col=1)
        fig.update_yaxes(title_text="Number of Anomalies", row=1, col=1)
        fig.update_yaxes(title_text="FOMC-Related Anomalies", row=2, col=1)
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"FOMC alignment chart saved to: {save_path}")
        
        return fig
    
    def create_dashboard(self, features_df: pd.DataFrame, anomalies_df: pd.DataFrame,
                        symbol: str, feature_importance: Dict = None) -> str:
        """
        Create a comprehensive HTML dashboard with all visualizations.
        
        Args:
            features_df: DataFrame with price data and features
            anomalies_df: DataFrame with anomaly detection results
            symbol: Stock symbol
            feature_importance: Dictionary of feature importance scores
        
        Returns:
            Path to saved dashboard HTML file
        """
        # Generate all visualizations
        price_fig = self.plot_price_with_anomalies(features_df, anomalies_df, symbol)
        distribution_fig = self.plot_anomaly_distribution(anomalies_df)
        method_fig = self.plot_method_comparison(anomalies_df)
        
        # Create HTML dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{symbol} - Anomaly Detection Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }}
                .dashboard-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #333;
                }}
                .chart-container {{
                    background-color: #2d2d2d;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }}
                h1, h2, h3 {{
                    color: #4ECDC4;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #333;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #FF6B6B;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #aaa;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header">
                    <h1>{symbol} - Anomalyy Dashboard</h1>
                    <p>Comprehensive analysis of market anomalies using multiple detection methods</p>
                </div>
                
                <!-- Metrics Summary -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">
                            {len(anomalies_df[anomalies_df['agreement_score'] >= 2]) if not anomalies_df.empty else 0}
                        </div>
                        <div class="metric-label">High-Confidence Anomalies</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">
                            {anomalies_df['agreement_score'].mean():.2f if not anomalies_df.empty else 0}
                        </div>
                        <div class="metric-label">Avg Agreement Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">
                            {len(features_df)}
                        </div>
                        <div class="metric-label">Trading Days Analyzed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">
                            {features_df['date'].min().strftime('%Y-%m-%d') if not features_df.empty else 'N/A'}
                        </div>
                        <div class="metric-label">Start Date</div>
                    </div>
                </div>
                
                <!-- Price Chart with Anomalies -->
                <div class="chart-container">
                    <h2>Price Chart with Detected Anomalies</h2>
                    <div id="price-chart"></div>
                </div>
                
                <!-- Anomaly Distribution -->
                <div class="chart-container">
                    <h2>Anomaly Type Distribution</h2>
                    <div id="anomaly-distribution"></div>
                </div>
                
                <!-- Method Comparison -->
                <div class="chart-container">
                    <h2>Detection Method Comparison</h2>
                    <div id="method-comparison"></div>
                </div>
                
                <!-- Feature Importance (if available) -->
                {self._create_feature_importance_html(feature_importance) if feature_importance else ''}
                
                <!-- Methodology Explanation -->
                <div class="chart-container">
                    <h2>Methodology</h2>
                    <p>This anomaly detection system uses three complementary methods:</p>
                    <ul>
                        <li><strong>Z-Score:</strong> Statistical method detecting extreme values in individual features</li>
                        <li><strong>Isolation Forest:</strong> Tree-based method isolating anomalies in feature space</li>
                        <li><strong>Local Outlier Factor (LOF):</strong> Density-based method comparing local density</li>
                    </ul>
                    <p>Anomalies are flagged when at least 2 out of 3 methods agree (high-confidence anomalies).</p>
                    <p><em>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                </div>
            </div>
            
            <script>
                // Load the Plotly figures
                const priceChart = {price_fig.to_json() if price_fig else '{}'};
                const anomalyDistribution = {distribution_fig.to_json() if distribution_fig else '{}'};
                const methodComparison = {method_fig.to_json() if method_fig else '{}'};
                
                // Render the charts
                if (priceChart.data) {{
                    Plotly.newPlot('price-chart', priceChart.data, priceChart.layout);
                }}
                if (anomalyDistribution.data) {{
                    Plotly.newPlot('anomaly-distribution', anomalyDistribution.data, anomalyDistribution.layout);
                }}
                if (methodComparison.data) {{
                    Plotly.newPlot('method-comparison', methodComparison.data, methodComparison.layout);
                }}
                
                // Add resize handler
                window.addEventListener('resize', function() {{
                    if (priceChart.data) Plotly.Plots.resize('price-chart');
                    if (anomalyDistribution.data) Plotly.Plots.resize('anomaly-distribution');
                    if (methodComparison.data) Plotly.Plots.resize('method-comparison');
                }});
            </script>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, f"{symbol}_dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Dashboard saved to: {dashboard_path}")
        return dashboard_path
    
    def _create_feature_importance_html(self, feature_importance: Dict) -> str:
        """Create HTML for feature importance visualization."""
        if not feature_importance:
            return ""
        
        # Get top 10 features
        top_features = list(feature_importance.items())[:10]
        
        # Create HTML table
        table_rows = ""
        for feature, importance in top_features:
            percentage = importance * 100
            table_rows += f"""
            <tr>
                <td>{feature}</td>
                <td>
                    <div style="background: linear-gradient(90deg, #4ECDC4 {percentage}%, #333 {percentage}%); 
                                padding: 5px; border-radius: 3px; color: white;">
                        {percentage:.1f}%
                    </div>
                </td>
            </tr>
            """
        
        return f"""
        <div class="chart-container">
            <h2>Top 10 Important Features for Anomaly Detection</h2>
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="border-bottom: 1px solid #444;">
                        <th style="text-align: left; padding: 10px;">Feature</th>
                        <th style="text-align: left; padding: 10px;">Importance</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """


if __name__ == "__main__":
    # Test visualization
    import yfinance as yf
    from feature_engineering import FeatureEngineer
    from anomaly_detection import AnomalyDetector
    
    print("Testing visualization...")
    
    # Fetch sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="3mo", interval="1d")
    df = df.reset_index()
    df = df.rename(columns={'Date': 'date'})
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(df)
    
    # Detect anomalies
    detector = AnomalyDetector(contamination=0.05)
    result_df = detector.detect_anomalies(features_df)
    
    # Create visualizations
    visualizer = Visualization("test_visualizations")
    
    # Price chart with anomalies
    fig1 = visualizer.plot_price_with_anomalies(features_df, result_df, "AAPL")
    visualizer.plot_price_with_anomalies(features_df, result_df, "AAPL", 
                                        save_path="test_visualizations/test_price_chart.html")
    
    # Anomaly distribution
    fig2 = visualizer.plot_anomaly_distribution(result_df)
    visualizer.plot_anomaly_distribution(result_df, 
                                        save_path="test_visualizations/test_anomaly_distribution.html")
    
    # Method comparison
    fig3 = visualizer.plot_method_comparison(result_df)
    visualizer.plot_method_comparison(result_df, 
                                     save_path="test_visualizations/test_method_comparison.html")
    
    # Create dashboard
    dashboard_path = visualizer.create_dashboard(features_df, result_df, "AAPL", 
                                                detector.feature_importance)
    
    print(f"Visualization test completed! Check the 'test_visualizations' directory.")
    print(f"Dashboard saved to: {dashboard_path}")
Visualizer = Visualization  # alias for compatibility
