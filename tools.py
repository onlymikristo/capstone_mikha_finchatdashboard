import os
import pandas as pd
import numpy as np
import datetime
import requests
from langchain_core.tools import tool


# --- API CONFIGURATION ---
SECTORS_API_BASE_URL = "https://api.sectors.app/v1"

# --- TOOL 1: Get Financial Data (REAL) ---


@tool
def get_financial_data(stock_ticker: str) -> dict:
    """
    Use this tool to get all essential financial and technical data for a 
    specific IDX stock ticker (e.g., "BBCA", "TLKM"). It returns key 
    support/resistance levels, SMAs, RSI, chart data, and the company's full name.
    """
    print(
        f"\n--- 🛠️ Calling REAL Tool: get_financial_data({stock_ticker}) ---\n")

    # FIX: Load environment and get key just-in-time
    from dotenv import load_dotenv
    load_dotenv()
    sectors_api_key = os.getenv("SECTORS_API_KEY")

    if not sectors_api_key:
        return {"error": "SECTORS_API_KEY is not configured. Cannot fetch real data."}

    try:
        headers = {"Authorization": sectors_api_key}

        # 1. Call the Company Report endpoint twice, once for each section.
        overview_response = requests.get(
            f"{SECTORS_API_BASE_URL}/company/report/{stock_ticker.lower()}/",
            headers=headers,
            params={"sections": "overview"}
        )
        overview_response.raise_for_status()
        overview_data = overview_response.json()

        valuation_response = requests.get(
            f"{SECTORS_API_BASE_URL}/company/report/{stock_ticker.lower()}/",
            headers=headers,
            params={"sections": "valuation"}
        )
        valuation_response.raise_for_status()
        valuation_data = valuation_response.json()

        # Extract data from the correct sections
        full_company_name = overview_data['company_name']
        pe_ratio = valuation_data['valuation']['forward_pe']
        market_cap_idr = overview_data['overview']['market_cap']

        # 2. Call the Daily Transaction Data endpoint for historical prices.
        end_date_str = datetime.date.today().isoformat()
        start_date_str = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        prices_response = requests.get(
            f"{SECTORS_API_BASE_URL}/daily/{stock_ticker.lower()}/",
            headers=headers,
            params={"start": start_date_str, "end": end_date_str}
        )
        prices_response.raise_for_status()
        prices_df = pd.DataFrame(prices_response.json())

        # --- START OF DATA PROCESSING (This part is complete) ---
        if prices_df.empty:
            return {"error": "No price data returned from API."}

        # This endpoint only provides 'close'. To avoid breaking the candlestick chart,
        # we will set open, high, and low to be the same as close.
        prices_df['open'] = prices_df['close']
        prices_df['high'] = prices_df['close']
        prices_df['low'] = prices_df['close']

        # Ensure date column is datetime
        prices_df['date'] = pd.to_datetime(prices_df['date'])

        # Calculate technicals
        prices_df['sma_50'] = prices_df['close'].rolling(window=50).mean()
        prices_df['sma_200'] = prices_df['close'].rolling(window=200).mean()

        delta = prices_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        prices_df['rsi'] = 100 - (100 / (1 + rs))

        # Get latest data
        latest = prices_df.iloc[-1]

        # Calculate Support & Resistance (simple version)
        support = prices_df['low'].rolling(window=50).min().iloc[-1]
        resistance = prices_df['high'].rolling(window=50).max().iloc[-1]
        # --- END OF DATA PROCESSING ---

        # --- Format the Output JSON (This is the final, structured output) ---
        return {
            "ticker": stock_ticker,
            "full_company_name": full_company_name,
            # FIX: Separate the bulky chart data from the agent-facing analytics.
            # The agent will use 'analytics_summary' for its reasoning.
            # It will pass 'chart_data_for_client' directly to the final output.
            "chart_data_for_client": prices_df.to_json(orient='records', date_format='iso'),
            "analytics_summary": {
                "sma_50": round(latest['sma_50'], 2) if latest['sma_50'] is not None else None,
                "sma_200": round(latest['sma_200'], 2) if latest['sma_200'] is not None and pd.notna(latest['sma_200']) else None,
                "rsi": round(latest['rsi'], 2) if latest['rsi'] is not None else None,
                "support": round(support, 2) if support is not None else None,
                "resistance": round(resistance, 2) if resistance is not None else None
            },
            "fundamentals": {
                "pe_ratio": round(pe_ratio, 2) if pe_ratio is not None else None,
                "market_cap_trillions_idr": round(market_cap_idr / 1e12, 2) if market_cap_idr is not None else None  # Convert to Trillions
            }
        }

    except Exception as e:
        print(f"Error in get_financial_data tool: {e}")
        return {"error": f"Failed to fetch data from sectors.app: {e}"}


# --- Tool 2: Dividend Info (REAL) ---
@tool
def get_dividend_info(stock_ticker: str) -> dict:
    """
    Use this tool when the user asks specifically about the stock's 
    dividend, dividend yield, or payout history.
    """
    print(
        f"\n--- 🛠️ Calling REAL Tool: get_dividend_info({stock_ticker}) ---\n")

    # FIX: Load environment and get key just-in-time
    from dotenv import load_dotenv
    load_dotenv()
    sectors_api_key = os.getenv("SECTORS_API_KEY")

    if not sectors_api_key:
        return {"error": "SECTORS_API_KEY is not configured."}

    try:
        headers = {"Authorization": sectors_api_key}
        # Call the Company Report endpoint for the Dividend section.
        response = requests.get(
            f"{SECTORS_API_BASE_URL}/company/report/{stock_ticker.lower()}/",
            headers=headers,
            params={"sections": "dividend"}
        )
        response.raise_for_status()
        dividend_data = response.json()['dividend']

        return {
            "dividend_yield_percent": dividend_data['yield'],
            "last_payout_idr": dividend_data['last_payout']
        }

    except Exception as e:
        print(f"Error in get_dividend_info tool: {e}")
        return {"error": f"Failed to fetch dividend data: {e}"}


# --- Compile all tools into a list ---
all_tools = [get_financial_data, get_dividend_info]
