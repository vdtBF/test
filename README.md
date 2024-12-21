# test

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
import re
import logging

    
logging.basicConfig(filename='stock_data_errors.log', level=logging.ERROR)

# List of stocks
tickers = ['AAPL', 'MSFT', 'TSLA', 'META', 'GOOG', 'GOOGL', 'AMD']

# Define start dates for different time periods
start_date = datetime(2024, 1, 1)  # For monthly closings since Jan 2024
start_date_1y = datetime.now() - timedelta(days=365)
start_date_2y = datetime.now() - timedelta(days=2*365)
start_date_3y = datetime.now() - timedelta(days=3*365)
start_date_5y = datetime.now() - timedelta(days=5*365)
start_date_10y = datetime.now() - timedelta(days=10*365)

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Fetch daily data for the last two days for daily change
        end_date = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)  # Market close time
        start_date_2d = end_date - timedelta(days=10)  # Look back 10 days to ensure we get at least 2 trading days
        daily_data = stock.history(start=start_date_2d, end=end_date)
        
        # Get the last two trading days for daily change
        trading_days = daily_data.index[-2:]  # Assumes we have at least 2 trading days in the last 10 days
        if len(trading_days) > 1:
            daily_change = (daily_data['Close'].iloc[-1] - daily_data['Close'].iloc[-2]) / daily_data['Close'].iloc[-2]
        else:
            daily_change = np.nan
            print(f"{ticker}: Not enough trading data for daily change calculation")

        def calculate_change(data, period_name):
            if not data.empty and len(data) > 1:
                print(f"{ticker}: {period_name} change calculated from {data.index[0].date()} (Price: ${data['Close'].iloc[0]:.2f}) to {data.index[-1].date()} (Price: ${data['Close'].iloc[-1]:.2f})")
                return (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            else:
                print(f"{ticker}: No data available for {period_name} change calculation")
                return np.nan
        
        # Fetch other historical data for different time periods
        weekly_data = stock.history(start=start_date, interval='1wk')
        monthly_data = stock.history(start=start_date, interval='1mo')
        history_1m = stock.history(period='1mo')
        history_3m = stock.history(period='3mo')
        history_6m = stock.history(period='6mo')
        history_1y = stock.history(start=start_date_1y)
        history_2y = stock.history(start=start_date_2y)
        history_3y = stock.history(start=start_date_3y)
        history_5y = stock.history(start=start_date_5y)
        history_10y = stock.history(start=start_date_10y)
        
        # Calculate returns
        weekly_change = calculate_change(weekly_data, 'weekly')
        monthly_change = calculate_change(history_1m, '1 month')
        three_month_change = calculate_change(history_3m, '3 months')
        six_month_change = calculate_change(history_6m, '6 months')
        one_year_change = calculate_change(history_1y, '1 year')
        two_year_change = calculate_change(history_2y, '2 years')
        three_year_change = calculate_change(history_3y, '3 years')
        five_year_change = calculate_change(history_5y, '5 years')
        ten_year_change = calculate_change(history_10y, '10 years')

        # Check for META and AMD on the first trading day after start_date_1y
        if ticker in ['META', 'AAPL']:
            next_trading_day = start_date_1y
            while True:
                stock_data = stock.history(start=next_trading_day, end=next_trading_day)
                if not stock_data.empty:
                    # We found a trading day
                    stock_data = stock_data.iloc[0]
                    print(f"\n{ticker} price data for {stock_data.name.date()}:")
                    for column, price in stock_data.items():
                        if column in ['Open', 'High', 'Low', 'Close']:
                            print(f"  - {column}: ${price:.2f}")
                        elif column == 'Volume':
                            print(f"  - {column}: {int(price):,} shares")
                    break
                else:
                    # Move to the next day
                    next_trading_day += timedelta(days=1)
                    if next_trading_day > datetime.now():
                        print(f"{ticker}: No trading data available after {start_date_1y.date()}")
                        break

        # Fetch the last weekly closing price
        last_week_data = stock.history(start=datetime.now() - timedelta(days=7), end=datetime.now(), interval='1wk')
        last_price = last_week_data['Close'].iloc[-1] if not last_week_data.empty else np.nan
        
        return {
            'Ticker': ticker,
            'Company Name': info.get('longName', ''),
            'Website': info.get('website', ''),
            'Closing Px': daily_data['Close'].iloc[-1] if not daily_data.empty else np.nan,
            'Last Px': last_price,
            'MC (B)': (info.get('sharesOutstanding', np.nan) / 1e6) * last_price / 1000 if pd.notna(info.get('sharesOutstanding', np.nan)) and pd.notna(last_price) else np.nan,
            'Shares Out': info.get('sharesOutstanding', np.nan),
            'Sector': info.get('sector', ''),
            'Industry': info.get('industry', ''),
            'Daily Δ': daily_change,
            'Weekly Δ': weekly_change,
            '1M Δ': monthly_change,
            '3M Δ': three_month_change,
            '6M Δ': six_month_change,
            '1Y Δ': one_year_change,
            '2Y Δ': two_year_change,
            '3Y Δ': three_year_change,
            '5Y Δ': five_year_change,
            '10Y Δ': ten_year_change,
            'Monthly Closings': monthly_data['Close'].tz_localize(None),
            'Info': info
        }
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return None

# Fetch data for specified tickers
all_data = [data for data in (fetch_stock_data(ticker) for ticker in tickers) if data]

# Create DataFrame for the main sheet
if all_data:  # Check if we have any data
    df_main = pd.DataFrame(all_data)
    df_main = df_main[['Ticker', 'Company Name', 'Website', 'Closing Px', 'Last Px', 'MC (B)', 
                       'Shares Out', 'Sector', 'Industry', 
                       'Daily Δ', 'Weekly Δ', '1M Δ', '3M Δ', '6M Δ', 
                       '1Y Δ', '2Y Δ', '3Y Δ', '5Y Δ', '10Y Δ']]
    
    # Calculate total Market Cap
    total_mcap = df_main['MC (B)'].sum()

    # Add new column 'Shares Out (M)'
    df_main['Shares Out (M)'] = df_main['Shares Out'] / 1e6

    # Add 'Weight' column
    df_main['Weight'] = df_main['MC (B)'] / total_mcap

    # Reorder columns to include new ones
    df_main = df_main[['Ticker', 'Company Name', 'Website', 'Sector', 'Industry', 'Closing Px',
                       'Last Px', 'MC (B)', 'Weight', 'Shares Out', 'Shares Out (M)', 
                       'Daily Δ', 'Weekly Δ', '1M Δ', '3M Δ', '6M Δ', 
                       '1Y Δ', '2Y Δ', '3Y Δ', '5Y Δ', '10Y Δ']]

    # Add total to DataFrame
    df_main.loc['Total'] = [np.nan] * len(df_main.columns)  # Use np.nan for numeric columns
    df_main.at['Total', 'MC (B)'] = total_mcap

    # Format numbers for main sheet
    for key in ['MC (B)']:
        df_main[key] = df_main[key].apply(lambda x: x if pd.notna(x) else np.nan)
    df_main['Shares Out (M)'] = df_main['Shares Out (M)'].apply(lambda x: x if pd.notna(x) else np.nan)
    for key in ['Closing Px', 'Last Px', 'Shares Out']:
        df_main[key] = df_main[key].apply(lambda x: x if pd.notna(x) else np.nan)
    for key in df_main.columns[13:]:  # All change columns, excluding 'Weight on Index'
        df_main[key] = df_main[key].apply(lambda x: x if pd.notna(x) else np.nan)

    # Format 'Weight' column
    df_main['Weight'] = df_main['Weight'].apply(lambda x: x if pd.notna(x) else np.nan)


    # Create a DataFrame for the Monthly Closings
    monthly_closings = pd.DataFrame()
    for data in all_data:
        monthly_data = pd.DataFrame({
            'Month': data['Monthly Closings'].index.to_series().dt.to_period('W-FRI').dt.strftime('%Y-%m-%d'),
            data['Ticker']: data['Monthly Closings'].values
        }).set_index('Month')
        monthly_closings = pd.concat([monthly_closings, monthly_data], axis=1)

    # Transpose for tickers on rows and months on columns
    monthly_closings = monthly_closings.T

    # Reset index to have tickers as a column
    monthly_closings = monthly_closings.reset_index().rename(columns={'index': 'Ticker'})

    # Format numbers for monthly closings table
    for column in monthly_closings.columns[1:]:  # Skip 'Ticker' column
        monthly_closings[column] = monthly_closings[column].apply(lambda x: x if pd.notna(x) else np.nan)




    # Create a DataFrame for Monthly Market Cap
    monthly_market_cap = pd.DataFrame()
    for data in all_data:
        shares_outstanding = data['Info'].get('sharesOutstanding', np.nan)
        if pd.notna(shares_outstanding):
            monthly_data = pd.DataFrame({
                'Month': data['Monthly Closings'].index.to_series().dt.to_period('W-FRI').dt.strftime('%Y-%m-%d'),
                data['Ticker']: [price * shares_outstanding / 1e6 for price in data['Monthly Closings']]  # Market Cap in millions
            }).set_index('Month')
            monthly_market_cap = pd.concat([monthly_market_cap, monthly_data], axis=1)

    # Transpose for tickers on rows and months on columns
    monthly_market_cap = monthly_market_cap.T

    # Reset index to have tickers as a column
    monthly_market_cap = monthly_market_cap.reset_index().rename(columns={'index': 'Ticker'})
    
    # Add totals to Monthly Market Cap
    monthly_market_cap.loc['Total'] = monthly_market_cap.sum(numeric_only=True)
    monthly_market_cap['Ticker'] = monthly_market_cap['Ticker'].fillna('Total')

    # Format numbers for monthly market cap table
    for column in monthly_market_cap.columns[1:]:  # Skip 'Ticker' column
        monthly_market_cap[column] = monthly_market_cap[column].apply(lambda x: x if pd.notna(x) else np.nan)




    # Create a DataFrame for Monthly Market Cap Weights
    monthly_mcap_weights = monthly_market_cap.copy()

    # Remove the total row before calculating weights
    monthly_mcap_weights = monthly_mcap_weights[monthly_mcap_weights['Ticker'] != 'Total']

    # Calculate weights for each month
    for column in monthly_mcap_weights.columns[1:]:  # Skip 'Ticker' column
        total_mcap = monthly_mcap_weights[column].sum()
        if total_mcap != 0:
            monthly_mcap_weights[column] = monthly_mcap_weights[column] / total_mcap
        else:
            monthly_mcap_weights[column] = np.nan

    # Add the total row back with the sum of weights
    monthly_mcap_weights.loc['Total'] = monthly_mcap_weights.sum(numeric_only=True)
    monthly_mcap_weights.at['Total', 'Ticker'] = 'Total'




    # Create a DataFrame for Sector Market Caps
    sector_market_cap = pd.DataFrame()

    # Group by sector and sum market caps
    for column in monthly_market_cap.columns[1:]:  # Skip 'Ticker' column
        sector_cap = monthly_market_cap[['Ticker', column]].merge(df_main[['Ticker', 'Sector']], on='Ticker').groupby('Sector')[column].sum().reset_index()
        sector_cap.columns = ['Sector', column]
        if sector_market_cap.empty:
            sector_market_cap = sector_cap
        else:
            sector_market_cap = sector_market_cap.merge(sector_cap, on='Sector', how='outer')

    # Ensure all sectors are included even if they don't have data for all months
    sectors = df_main['Sector'].unique()
    sector_market_cap = sector_market_cap.set_index('Sector').reindex(sectors).reset_index().fillna(0)  # Fill NaN with 0 for sectors with no data

    # Convert 'Sector' to string to avoid type mismatch
    sector_market_cap['Sector'] = sector_market_cap['Sector'].astype(str)

    # Sort by sector name for consistency
    sector_market_cap = sector_market_cap.sort_values('Sector')

    # Add total to DataFrame
    sector_market_cap.loc['Total'] = sector_market_cap.sum(numeric_only=True)
    sector_market_cap['Sector'] = sector_market_cap['Sector'].fillna('Total')

    # Create a DataFrame for Sector Market Cap Weights
    sector_mcap_weights = sector_market_cap.copy()

    # Remove the total row before calculating weights
    sector_mcap_weights = sector_mcap_weights[sector_mcap_weights['Sector'] != 'Total']

    # Calculate weights for each month
    for column in sector_mcap_weights.columns[1:]:  # Skip 'Sector' column
        total_mcap = sector_mcap_weights[column].sum()
        if total_mcap != 0:
            sector_mcap_weights[column] = sector_mcap_weights[column] / total_mcap
        else:
            sector_mcap_weights[column] = np.nan

    # Add the total row back with the sum of weights
    sector_mcap_weights.loc['Total'] = sector_mcap_weights.sum(numeric_only=True)
    sector_mcap_weights.at['Total', 'Sector'] = 'Total'

    # Create a DataFrame for Merged Market Cap by Company
    merged_market_cap = pd.DataFrame()

    # Dictionary to hold companies and their tickers
    companies = {}

    # Group tickers by company name
    for data in all_data:
        company_name = data['Company Name']
        if company_name not in companies:
            companies[company_name] = []
        companies[company_name].append(data['Ticker'])

    # Process each company
    for company_name, tickers in companies.items():
        if len(tickers) > 1:  # Only process when there's more than one ticker
            # Combine market caps for this company
            company_market_caps = monthly_market_cap[monthly_market_cap['Ticker'].isin(tickers)].set_index('Ticker')
        
            # Merge market caps
            merged_cap = company_market_caps.sum()

            # Create a new row for this merged company
            new_row = pd.DataFrame({
                'Ticker': [f"{tickers[0]} + {tickers[1]}"],  # Assuming only two tickers per company for simplicity
                **{month: [cap] for month, cap in merged_cap.items()}
            })
            merged_market_cap = pd.concat([merged_market_cap, new_row], ignore_index=True)
        else:
            # If there's only one ticker, just add it as is
            single_ticker_data = monthly_market_cap[monthly_market_cap['Ticker'] == tickers[0]]
            merged_market_cap = pd.concat([merged_market_cap, single_ticker_data], ignore_index=True)

    # Add totals to Merged Market Cap
    merged_market_cap.loc['Total'] = merged_market_cap.sum(numeric_only=True)
    merged_market_cap['Ticker'] = merged_market_cap['Ticker'].fillna('Total')

    # Format numbers for merged market cap table
    for column in merged_market_cap.columns[1:]:  # Skip 'Ticker' column
        merged_market_cap[column] = merged_market_cap[column].apply(lambda x: x if pd.notna(x) else np.nan)



    # Save to Excel file
    save_folder = r'C:\Users\User\Desktop\Stocks\Python'  # Replace with your actual path
    file_name = 'SPY_v1.xlsx'
    full_path = os.path.join(save_folder, file_name)

    with pd.ExcelWriter(full_path, mode='w', engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Data', index=False)
        monthly_closings.to_excel(writer, sheet_name='Monthly Px', index=False)
        monthly_market_cap.to_excel(writer, sheet_name='Monthly MC', index=False)
        monthly_mcap_weights.to_excel(writer, sheet_name='MC Weights', index=False)
        sector_market_cap.to_excel(writer, sheet_name='Sector MC', index=False)
        sector_mcap_weights.to_excel(writer, sheet_name='Sector MCw', index=False)
        merged_market_cap.to_excel(writer, sheet_name='Merged MC', index=False)

    

    print(f"Data has been saved to {full_path}")
else:
    print("No data was fetched for any of the stocks.")
