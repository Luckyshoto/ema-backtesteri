import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="EMA Crossover Backtester",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 EMA Crossover Backtester")
st.markdown("""
This app backtests an **EMA (Exponential Moving Average) Crossover Strategy**.
- **Buy** when Fast EMA crosses above Slow EMA
- **Sell** when Fast EMA crosses below Slow EMA
""")

# Sidebar for user inputs
st.sidebar.header("Strategy Parameters")

# User inputs
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    symbol = st.text_input("Stock Symbol", "AAPL").upper()
with col2:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
with col3:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

col4, col5 = st.sidebar.columns(2)
with col4:
    fast_ema = st.number_input("Fast EMA Period", min_value=5, max_value=100, value=20)
with col5:
    slow_ema = st.number_input("Slow EMA Period", min_value=10, max_value=200, value=50)

initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000, value=10000)

# Add transaction cost option
transaction_cost = st.sidebar.number_input("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1) / 100

# Strategy class
class EMAStrategy:
    def __init__(self, data, fast_ema, slow_ema):
        self.data = data.copy()
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.results = None
    
    def calculate_indicators(self):
        """Calculate EMA indicators"""
        self.data['EMA_fast'] = self.data['Close'].ewm(span=self.fast_ema).mean()
        self.data['EMA_slow'] = self.data['Close'].ewm(span=self.slow_ema).mean()
        
        # Generate signals
        self.data['Signal'] = np.where(self.data['EMA_fast'] > self.data['EMA_slow'], 1, 0)
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data
    
    def run_backtest(self, initial_capital=10000, transaction_cost=0.001):
        """Run the backtest with transaction costs"""
        self.calculate_indicators()
        
        # Calculate returns
        self.data['Market_Return'] = self.data['Close'].pct_change()
        
        # Strategy returns (accounting for transaction costs)
        self.data['Strategy_Return'] = self.data['Signal'].shift(1) * self.data['Market_Return']
        
        # Apply transaction costs when positions change
        trade_days = self.data[self.data['Position'] != 0].index
        for trade_day in trade_days:
            self.data.loc[trade_day, 'Strategy_Return'] -= transaction_cost
        
        # Calculate cumulative returns
        self.data['Cumulative_Market'] = (1 + self.data['Market_Return']).cumprod()
        self.data['Cumulative_Strategy'] = (1 + self.data['Strategy_Return']).cumprod()
        
        # Portfolio values
        self.data['Portfolio_Value'] = initial_capital * self.data['Cumulative_Strategy']
        self.data['Market_Value'] = initial_capital * self.data['Cumulative_Market']
        
        # Calculate drawdown
        self.data['Strategy_Peak'] = self.data['Portfolio_Value'].cummax()
        self.data['Strategy_Drawdown'] = (self.data['Portfolio_Value'] - self.data['Strategy_Peak']) / self.data['Strategy_Peak']
        
        self.data['Market_Peak'] = self.data['Market_Value'].cummax()
        self.data['Market_Drawdown'] = (self.data['Market_Value'] - self.data['Market_Peak']) / self.data['Market_Peak']
        
        self.results = self.data
        return self.data
    
    def get_performance_metrics(self, initial_capital):
        """Calculate performance metrics"""
        if self.results is None:
            return {}
            
        final_portfolio = self.results['Portfolio_Value'].iloc[-1]
        final_market = self.results['Market_Value'].iloc[-1]
        
        total_return = (final_portfolio - initial_capital) / initial_capital * 100
        market_return = (final_market - initial_capital) / initial_capital * 100
        
        # Annualized return
        days = len(self.results)
        years = days / 252
        annualized_return = ((final_portfolio / initial_capital) ** (1/years) - 1) * 100
        market_annualized = ((final_market / initial_capital) ** (1/years) - 1) * 100
        
        # Volatility
        volatility = self.results['Strategy_Return'].std() * np.sqrt(252) * 100
        market_volatility = self.results['Market_Return'].std() * np.sqrt(252) * 100
        
        # Sharpe Ratio
        sharpe = annualized_return / volatility if volatility != 0 else 0
        market_sharpe = market_annualized / market_volatility if market_volatility != 0 else 0
        
        # Max Drawdown
        max_drawdown = self.results['Strategy_Drawdown'].min() * 100
        market_max_drawdown = self.results['Market_Drawdown'].min() * 100
        
        # Win rate
        winning_trades = (self.results['Strategy_Return'] > 0).sum()
        total_trades = (self.results['Position'] != 0).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        metrics = {
            'Final Portfolio Value': final_portfolio,
            'Final Market Value': final_market,
            'Total Return (%)': total_return,
            'Market Return (%)': market_return,
            'Annualized Return (%)': annualized_return,
            'Market Annualized (%)': market_annualized,
            'Volatility (%)': volatility,
            'Market Volatility (%)': market_volatility,
            'Sharpe Ratio': sharpe,
            'Market Sharpe': market_sharpe,
            'Max Drawdown (%)': max_drawdown,
            'Market Max Drawdown (%)': market_max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': total_trades
        }
        
        return metrics

# Main app
def main():
    # Download data
    @st.cache_data
    def load_data(symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
            return data
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return None
    
    # Run backtest button
    if st.sidebar.button("Run Backtest", type="primary"):
        with st.spinner("Downloading data and running backtest..."):
            # Load data
            data = load_data(symbol, start_date, end_date)
            
            if data is not None:
                # Initialize and run strategy
                strategy = EMAStrategy(data, fast_ema, slow_ema)
                results = strategy.run_backtest(initial_capital, transaction_cost)
                metrics = strategy.get_performance_metrics(initial_capital)
                
                # Display results
                st.header("📊 Backtest Results")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Strategy Return", 
                        f"{metrics['Total Return (%)']:.2f}%",
                        f"{metrics['Total Return (%)'] - metrics['Market Return (%)']:.2f}% vs Market"
                    )
                
                with col2:
                    st.metric(
                        "Final Portfolio Value", 
                        f"${metrics['Final Portfolio Value']:,.2f}",
                        f"${metrics['Final Portfolio Value'] - metrics['Final Market Value']:,.2f} vs Market"
                    )
                
                with col3:
                    st.metric(
                        "Annualized Return", 
                        f"{metrics['Annualized Return (%)']:.2f}%",
                        f"{metrics['Annualized Return (%)'] - metrics['Market Annualized (%)']:.2f}% vs Market"
                    )
                
                with col4:
                    st.metric(
                        "Sharpe Ratio", 
                        f"{metrics['Sharpe Ratio']:.2f}",
                        f"{metrics['Sharpe Ratio'] - metrics['Market Sharpe']:.2f} vs Market"
                    )
                
                # Detailed metrics
                st.subheader("Detailed Performance Metrics")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.write("**Strategy Metrics:**")
                    st.write(f"Max Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
                    st.write(f"Volatility: {metrics['Volatility (%)']:.2f}%")
                    st.write(f"Win Rate: {metrics['Win Rate (%)']:.2f}%")
                    st.write(f"Total Trades: {metrics['Total Trades']}")
                
                with metrics_col2:
                    st.write("**Market Metrics:**")
                    st.write(f"Market Return: {metrics['Market Return (%)']:.2f}%")
                    st.write(f"Market Max Drawdown: {metrics['Market Max Drawdown (%)']:.2f}%")
                    st.write(f"Market Volatility: {metrics['Market Volatility (%)']:.2f}%")
                    st.write(f"Market Sharpe: {metrics['Market Sharpe']:.2f}")
                
                # Create interactive plots
                st.subheader("Charts")
                
                # Chart 1: Price and EMA with signals
                fig1 = make_subplots(rows=2, cols=1, 
                                   subplot_titles=('Price and EMA Crossover', 'Portfolio Value'),
                                   vertical_spacing=0.1,
                                   row_heights=[0.6, 0.4])
                
                # Price and EMA
                fig1.add_trace(
                    go.Scatter(x=results.index, y=results['Close'], 
                              name='Price', line=dict(color='blue')),
                    row=1, col=1
                )
                fig1.add_trace(
                    go.Scatter(x=results.index, y=results['EMA_fast'], 
                              name=f'EMA {fast_ema}', line=dict(color='orange')),
                    row=1, col=1
                )
                fig1.add_trace(
                    go.Scatter(x=results.index, y=results['EMA_slow'], 
                              name=f'EMA {slow_ema}', line=dict(color='red')),
                    row=1, col=1
                )
                
                # Buy/Sell signals
                buy_signals = results[results['Position'] == 1]
                sell_signals = results[results['Position'] == -1]
                
                fig1.add_trace(
                    go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                              mode='markers', name='Buy',
                              marker=dict(color='green', size=8, symbol='triangle-up')),
                    row=1, col=1
                )
                fig1.add_trace(
                    go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                              mode='markers', name='Sell',
                              marker=dict(color='red', size=8, symbol='triangle-down')),
                    row=1, col=1
                )
                
                # Portfolio value
                fig1.add_trace(
                    go.Scatter(x=results.index, y=results['Portfolio_Value'],
                              name='Strategy', line=dict(color='green')),
                    row=2, col=1
                )
                fig1.add_trace(
                    go.Scatter(x=results.index, y=results['Market_Value'],
                              name='Buy & Hold', line=dict(color='blue', dash='dash')),
                    row=2, col=1
                )
                
                fig1.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 2: Drawdown
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(x=results.index, y=results['Strategy_Drawdown'] * 100,
                              name='Strategy Drawdown', fill='tozeroy', line=dict(color='red'))
                )
                fig2.add_trace(
                    go.Scatter(x=results.index, y=results['Market_Drawdown'] * 100,
                              name='Market Drawdown', fill='tozeroy', line=dict(color='blue'))
                )
                fig2.update_layout(title='Drawdown Comparison',
                                 xaxis_title='Date',
                                 yaxis_title='Drawdown (%)',
                                 height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show raw data
                st.subheader("Raw Data")
                st.dataframe(results.tail(20))
                
                # Download results
                csv = results.to_csv()
                st.download_button(
                    label="Download Backtest Results as CSV",
                    data=csv,
                    file_name=f"ema_backtest_{symbol}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # Instructions
    else:
        st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Backtest' to start!")
        
        # Educational content
        st.markdown("""
        ### 📚 How EMA Crossover Works
        
        **Strategy Rules:**
        - **BUY Signal**: When Fast EMA (${fast_ema}) crosses above Slow EMA (${slow_ema})
        - **SELL Signal**: When Fast EMA crosses below Slow EMA
        
        **Key Concepts:**
        - **EMA (Exponential Moving Average)**: Gives more weight to recent prices
        - **Fast EMA**: Reacts quickly to price changes (shorter period)
        - **Slow EMA**: Reacts slowly to price changes (longer period)
        - **Crossover**: When the two EMA lines cross, indicating potential trend changes
        
        **Common EMA Periods:**
        - Short-term: 9, 12, 20
        - Long-term: 21, 26, 50
        
        **Note**: This is for educational purposes. Past performance doesn't guarantee future results.
        """)

if __name__ == "__main__":
    main()