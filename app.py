import streamlit as st
import json
import pandas as pd
import re
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables at the very start of the app
load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage

# Import our agent executor from finai_agent.py
from finai_agent import agent_executor

# --- Helper Function to Render a Simple Line Chart (Google Style) ---


def render_simple_line_chart(chart_data: dict):
    """Parses the agent's chart data and renders a simple line chart."""

    # FIX: The chart_data is a list of records (dicts).
    # We need to check if it's a non-empty list and if the first record has the keys we need.
    if not isinstance(chart_data, list) or not chart_data or not all(k in chart_data[0] for k in ['date', 'close']):
        st.warning("Chart data is in an unexpected format.")
        return None

    df = pd.DataFrame(chart_data)
    df['date'] = pd.to_datetime(df['date'])
    # FIX: Create a pre-formatted text column for the hover label
    df['formatted_price'] = df['close'].apply(format_rupiah)

    fig = go.Figure()

    # Add the main price line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='lines', name='Close Price',
        line=dict(color='#1f77b4', width=2),
        # FIX: Use the pre-formatted text in the custom hover template
        hovertemplate='%{x|%b %d, %Y}<br><b>%{customdata}</b><extra></extra>',
        customdata=df['formatted_price']
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title=None,
        yaxis_title="Price (IDR)",
        xaxis_rangeslider_visible=False,  # Clean look like Google
        height=300,  # A more compact height
        margin=dict(l=20, r=20, t=30, b=20)  # Tight margins
    )
    return fig

# --- Helper Function for Currency Formatting ---

def format_rupiah(price: float) -> str:
    """Formats a float into proper Indonesian Rupiah accounting style (e.g., Rp 7.250,00)."""
    if price is None:
        return "Rp 0,00"
    # Format with a temporary placeholder for thousands and comma for decimal
    return f"Rp {price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def post_process_narrative(narrative: str) -> str:
    """Finds all currency values in the narrative and formats them."""
    # FIX: This regex is now more robust. It correctly captures integers and decimals,
    # handles existing separators, and prevents incorrect re-formatting.
    def replacer(match):
        try:
            # Remove thousand separators (dots), change decimal comma to dot, then convert to float.
            number_str = match.group(1).replace(".", "").replace(",", ".")
            number = float(number_str)
            return format_rupiah(number)
        except (ValueError, IndexError):
            return match.group(0) # Return original match if conversion fails
    return re.sub(r'Rp\s*([\d,.]+\d)', replacer, narrative)
# --- Helper Function to Render the Dashboard-in-a-Bubble ---


def render_dashboard_response(response_json: dict, user_query: str):
    """Parses the agent's JSON and renders the full dashboard."""

    # Try to parse the user's query for their price
    user_price = 0.0
    try:
        # A simple parser
        words = user_query.replace(".", "").replace(",", "").split()
        if "rp" in words:
            user_price = float(words[words.index("rp") + 1])
        elif "at" in words:
            user_price = float(words[words.index("at") + 1])
    except:
        pass  # Failed to parse price, just use 0.0

    # FIX: The response is now a single dictionary, not wrapped in a list.
    stock_data = response_json
    ticker = stock_data.get('ticker', 'N/A')

    # --- 1. AI Financial Analysis (Text) ---
    narrative = stock_data.get('narrative_summary', 'No summary available.')
    # FIX: Apply currency formatting to the narrative text
    formatted_narrative = post_process_narrative(narrative)
    st.markdown(formatted_narrative)

    # --- NEW: Simple Price Chart ---
    chart_data_str = stock_data.get('chart_data_json')
    if chart_data_str and chart_data_str != '{}':
        try:
            chart_data = json.loads(chart_data_str)
            fig = render_simple_line_chart(chart_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render chart: {e}")

    # --- 2. Trading Plan Cards ---
    buy_price_str = stock_data.get('buy_target_price', 'Rp 0')
    sell_price_str = stock_data.get('sell_target_price', 'Rp 0')

    # Helper to convert "Rp 53" or "Rp 9.250" to a float
    def price_to_float(price_str: str) -> float:
        try:
            # FIX: This handles both "Rp 7300" and "Rp 7.300" by removing all non-digit characters
            # (except a potential decimal point, though not expected here).
            return float(re.sub(r'[^\d.]', '', price_str))
        except (ValueError, AttributeError, TypeError):
            return 0.0

    col1, col2 = st.columns(2)
    with col1:
        buy_price = price_to_float(buy_price_str)
        st.metric(label=f"BUY Target ({ticker})", value=format_rupiah(buy_price))
        st.markdown(f"*{stock_data.get('buy_target_date_estimate', 'N/A')}*")

    with col2:
        sell_price = price_to_float(sell_price_str)
        st.metric(label=f"SELL Target ({ticker})", value=format_rupiah(sell_price))
        st.markdown(f"*{stock_data.get('sell_target_date_estimate', 'N/A')}*")

    # --- 4. Key Metrics ---
    st.divider()
    st.subheader("Key Financial Metrics")
    fundamentals = stock_data.get('fundamentals', {})
    pe_ratio = fundamentals.get('pe_ratio', 0)
    market_cap = fundamentals.get('market_cap_trillions_idr', 0)

    col3, col4 = st.columns(2)
    with col3:
        # FIX: Check for None before formatting to prevent crashes.
        pe_value = f"{pe_ratio:.2f}x" if pe_ratio is not None else "N/A"
        st.metric(label="P/E Ratio", value=pe_value)
    with col4:
        # FIX: Check for None before formatting.
        mc_value = f"Rp {market_cap:.2f} T" if market_cap is not None else "N/A"
        st.metric(label="Market Cap (IDR)",
                    value=mc_value)


# --- Main App ---


# 1. Set Page Config (Wide 16:9 Layout)
st.set_page_config(layout="wide", page_title="FinAI IDX Stock Advisor")
st.title("FinAI IDX Stock Advisor 🇮🇩")

# 2. Initialize Session State for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I am your AI-powered IDX analyst. How can I help you today? (e.g., 'Analyze my position in a stock. I bought at Rp 9,250')")
    ]

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg.type):
        # FIX: This is the permanent fix for the 'str object has no attribute get' error.
        # We check if the content is a valid JSON string before trying to parse and render it.
        is_dashboard = msg.content.startswith('{') and msg.content.endswith('}')
        if msg.type == "ai" and is_dashboard:
            try:
                data = json.loads(msg.content)
                # Find the corresponding user query that triggered this dashboard
                user_query = ""
                current_index = st.session_state.messages.index(msg)
                if current_index > 0:
                    user_query = st.session_state.messages[current_index - 1].content
                render_dashboard_response(data, user_query)
            except (json.JSONDecodeError, Exception) as e:
                st.error(f"An error occurred while rendering this message: {e}")
                st.markdown(msg.content) # Show raw content on failure
        else:
            # It's a user message or a simple text response from the AI
            # FIX: Apply currency formatting to ALL AI text responses for consistency.
            formatted_content = post_process_narrative(msg.content) if msg.type == 'ai' else msg.content
            st.markdown(formatted_content)

# 4. Handle User Input
if prompt := st.chat_input("Analyze BBCA, compare BBCA and BBRI, or ask a question..."):

    # Add user message to history and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Call the Agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            # Invoke the agent
            try:
                result = agent_executor.invoke({
                    "input": prompt,
                    # Pass the history *except* the new prompt
                    "chat_history": st.session_state.messages[:-1]
                })

                # The result can be a dict (for dashboard) or a str (for text).
                # If it's a dict, convert it to a JSON string for storage in session state.
                if isinstance(result, dict):
                    response_content = json.dumps(result)
                else:
                    # It's already a string, so just use it directly.
                    response_content = str(result)

            except Exception as e:
                # Display the actual error message in the chat for debugging purposes.
                error_message = f"Sorry, I encountered an error while trying to process your request:\n\n```\n{e}\n```"
                st.error(error_message) # Show the error in a prominent red box
                response_content = error_message # Also put the error in the chat bubble

            # Add the full AI response (JSON or text) to history
            st.session_state.messages.append(
                AIMessage(content=response_content))

            # Trigger a rerun to display the new message from the history.
            st.rerun()
