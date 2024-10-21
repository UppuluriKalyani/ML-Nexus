import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

def implement_trading_strategy(predictions, actual_prices, initial_balance=10000):
    balance = initial_balance
    shares = 0
    trade_log = []
    profit_log = []

    for i in range(1, len(predictions)):
        current_price = actual_prices[i]
        
        if predictions[i] > predictions[i-1] and balance > current_price:
            # Buy as many shares as possible
            shares_to_buy = balance // current_price
            cost = shares_to_buy * current_price
            balance -= cost
            shares += shares_to_buy
            trade_log.append(('Buy', actual_prices.index[i], current_price, shares_to_buy))
        elif predictions[i] < predictions[i-1] and shares > 0:
            # Sell all shares
            balance += shares * current_price
            trade_log.append(('Sell', actual_prices.index[i], current_price, shares))
            shares = 0

        total_value = balance + shares * current_price
        profit_log.append(total_value)

    final_balance = balance + shares * actual_prices[-1]
    total_profit = final_balance - initial_balance
    return trade_log, profit_log, final_balance, total_profit

def save_results(trade_log, profit_log, final_balance, total_profit, initial_balance):
    # Save trade log
    df_trade_log = pd.DataFrame(trade_log, columns=['Action', 'Date', 'Price', 'Shares'])
    df_trade_log.to_csv('results/trade_log.csv', index=False)

    # Save performance metrics
    with open('results/performance_metrics.txt', 'w') as f:
        f.write(f"Initial Balance: ${initial_balance:.2f}\n")
        f.write(f"Final Balance: ${final_balance:.2f}\n")
        f.write(f"Total Profit: ${total_profit:.2f}\n")
        f.write(f"Return: {(total_profit / initial_balance) * 100:.2f}%\n")

def plot_results(actual_prices, profit_log, trade_log):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Stock price with buy/sell signals
    ax1.plot(actual_prices.index, actual_prices, label='Actual Prices', color='blue')
    for action, date, price, shares in trade_log:
        if action == 'Buy':
            ax1.scatter(date, price, marker='^', color='green', s=100)
        elif action == 'Sell':
            ax1.scatter(date, price, marker='v', color='red', s=100)
    ax1.set_title('Stock Price with Buy/Sell Signals')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Plot 2: Portfolio value over time
    ax2.plot(actual_prices.index[1:], profit_log, label='Portfolio Value', color='purple')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value ($)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/trading_results.png')
    plt.close()

if __name__ == "__main__":
    # Load data
    data = load_data('data/AAPL_data.csv')
    test_predictions = np.loadtxt('data/test_predictions.csv', delimiter=',')

    # Implement trading strategy
    initial_balance = 10000
    trade_log, profit_log, final_balance, total_profit = implement_trading_strategy(
        test_predictions, data['Close'][-len(test_predictions):], initial_balance
    )

    # Save results
    save_results(trade_log, profit_log, final_balance, total_profit, initial_balance)

    # Plot results
    plot_results(data['Close'][-len(test_predictions):], profit_log, trade_log)

    print("Trading bot simulation complete. Results saved in the 'results' directory.")