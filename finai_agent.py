import os
import httpx
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# Import our custom tools
from tools import all_tools

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("FATAL ERROR: GROQ_API_KEY not found in .env file.")
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

# --- PROXY FIX ---
try:
    sync_client = httpx.Client(trust_env=False)
    async_client = httpx.AsyncClient(trust_env=False)
except TypeError:
    print("Warning: httpx version doesn't support trust_env. Unsetting proxy variables.")
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('http_proxy', None)
    os.environ.pop('httpss_proxy', None)
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
except Exception as e:
    print(f"Error initializing httpx clients: {e}")
    sync_client = None
    async_client = None


# --- 1. DEFINE PROMPTS FOR DIFFERENT TASKS ---

# Router Prompt: Classifies the user's intent.
router_prompt_template = """
Given the user's input and chat history, classify the user's intent into one of the following categories:
'stock_analysis', 'stock_comparison', 'dividend_question', 'general_conversation'.

Chat History:
{chat_history}

User Input:
{input}

Classification:"""

# Single Stock Analysis Prompt: The one we've already refined.
synthesis_prompt_template = """
You are FinAI, an expert Indonesian stock market analyst. Your task is to synthesize the provided data into a clear, data-driven trading plan.

**User's Query:**
{input}

**Financial Data:**
{analytics}

Based on all the data above, generate a trading plan.
- The `narrative_summary` should be insightful and directly reference the provided financial data.
- If the user provided their buy price/date, the `buy_target_price` should reflect the user's entry price, and the `buy_target_date_estimate` should be a note like "User's entry on Oct 1st".
- If the user did NOT provide a buy price, you should suggest a `buy_target_price` and provide a rationale in the `buy_target_date_estimate` (e.g., "Entry near support level").
- The `sell_target_date_estimate` MUST always contain a forward-looking estimate and a brief rationale for the `sell_target_price` (e.g., "Within 1-2 months, approaching resistance").
- The `buy_target_price` and `sell_target_price` should be realistic estimates based on the support and resistance levels in the financial data.
- You MUST return only a single, valid JSON object and nothing else. Your output must conform to the following structure:
{{
    "ticker": "...",
    "narrative_summary": "...",
    "buy_target_price": "Rp <price>", // e.g., "Rp 53" or "Rp 9250"
    "buy_target_date_estimate": "...",
    "sell_target_price": "Rp <price>", // e.g., "Rp 62" or "Rp 10000"
    "sell_target_date_estimate": "..."
}}
"""

# Stock Comparison Prompt
comparison_prompt_template = """
You are FinAI, an expert Indonesian stock market analyst. Your task is to provide a concise comparison of the two stocks based on the data provided.
Focus on key differences in their P/E ratio, market cap, and recent performance (RSI, SMAs).

**User's Query:**
{input}

**Data for {ticker_1}:**
{data_1}

**Data for {ticker_2}:**
{data_2}

Provide your analysis as a conversational markdown text. Do not return JSON.
"""

# General Q&A Prompt
qa_prompt_template = """
You are FinAI, a helpful financial analyst assistant. Answer the user's question based on the chat history.
If the question is about a stock you have just analyzed, use the information from the history.

Chat History:
{chat_history}

User Input:
{input}

Answer:"""

# Initialize the LLM (Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
    http_client=sync_client,
    http_async_client=async_client
)

# --- 2. DEFINE HELPER FUNCTIONS AND PARSERS ---

# Unpack the tools for easier access
get_financial_data, get_dividend_info = all_tools

def parse_ticker_from_input(x: dict) -> str:
    """A simple parser to find the first likely stock ticker in the user's input."""
    # FIX: The regex was failing if a ticker was followed by punctuation (e.g., "TLKM?").
    # This new regex looks for a 4-letter word and is not stopped by a trailing question mark.
    match = re.search(r'\b([A-Z]{4})\b', x["input"].upper().replace('?', ''))
    if match:
        ticker = match.group(1)
        return ticker
    # If no ticker is found in a specific query, it might be a general question.
    # Return a neutral value that won't trigger an invalid response.
    return "GENERAL"

def parse_two_tickers(x: dict) -> dict:
    """A simple parser to find two stock tickers for comparison."""
    # FIX: This logic is now robust. It finds unique tickers and only adds defaults if needed.
    # This prevents the agent from trying to compare a stock with itself, which causes hallucinations.
    found_tickers = list(dict.fromkeys(re.findall(r'\b[A-Z]{4}\b', x["input"].upper()))) # Find unique tickers
    default_tickers = ["BBCA", "BBRI"]
    while len(found_tickers) < 2:
        found_tickers.append(default_tickers.pop(0))
    return {
        "ticker_1": found_tickers[0],
        "ticker_2": found_tickers[1],
        "input": x["input"]
    }


# --- 3. DEFINE SPECIALIZED CHAINS FOR EACH TASK ---

# --- Chain 1: Single Stock Analysis ---

# Data-gathering part
# This chain now takes the full context, adds the ticker and financials to the 'passthrough' key.
data_gathering_chain = (
    RunnablePassthrough.assign(
        passthrough=lambda x: x['passthrough'] | {'ticker': parse_ticker_from_input(x['passthrough'])}
    )
    | RunnablePassthrough.assign(
        passthrough=lambda x: x['passthrough'] | {'financials': get_financial_data.invoke(x['passthrough']['ticker'])}
    )
)

# Prepares the small context for the LLM
# This now correctly reads from the 'passthrough' object.
prepare_for_synthesis = RunnableParallel(
    input=lambda x: x["passthrough"]["input"],
    analytics=lambda x: x["passthrough"]["financials"]["analytics_summary"]
)

# Synthesis part
synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
synthesis_chain = RunnableParallel(
    llm_analysis=prepare_for_synthesis | synthesis_prompt | llm | JsonOutputParser(),
    chart_data_json=lambda x: x["passthrough"]["financials"]["chart_data_for_client"],
    fundamentals=lambda x: x["passthrough"]["financials"]["fundamentals"]
)

# Final formatting step
def format_analysis_output(result_json: dict) -> dict:
    """Merges the LLM analysis with the passthrough data."""
    # FIX: Return a single, clean dictionary instead of a wrapped one.
    # This makes the chat history much cleaner for the router.
    final_data = result_json['llm_analysis']
    final_data['chart_data_json'] = result_json['chart_data_json']
    final_data['fundamentals'] = result_json['fundamentals']
    return final_data

# The complete single-stock analysis chain
stock_analysis_chain = data_gathering_chain | RunnableBranch(
    # Condition: Check if the 'financials' key inside 'passthrough' contains an error.
    (lambda x: "error" in x.get("passthrough", {}).get("financials", {}),
     # If True: Return a user-friendly error message.
     RunnableLambda(lambda x: f"Sorry, I could not retrieve data for the ticker '{x['passthrough'].get('ticker', '')}'. The API returned an error: {x['passthrough']['financials']['error']}")),
    # If False (default): Proceed with the normal analysis flow.
    synthesis_chain | format_analysis_output
)


# --- Chain 2: Stock Comparison ---

comparison_data_chain = (
    (lambda x: x['passthrough']) 
    | RunnableLambda(parse_two_tickers)
) | RunnableLambda(
    # This structure ensures that the parallel tool calls are executed,
    # and their results are merged back with the original input dictionary.
    lambda x: RunnablePassthrough.assign(
        data_1=lambda y: get_financial_data.invoke(y['ticker_1']),
        data_2=lambda y: get_financial_data.invoke(y['ticker_2']),
    ).invoke(x)
)

comparison_prompt = ChatPromptTemplate.from_template(comparison_prompt_template)
stock_comparison_chain = comparison_data_chain | RunnableLambda(
    # Only run the LLM part if we have data, not the error string
    lambda x: (comparison_prompt | llm | StrOutputParser()) if isinstance(x, dict) else x
)


# --- Chain 3: General Conversation ---

qa_prompt = ChatPromptTemplate.from_template(qa_prompt_template)
general_conversation_chain = (lambda x: x['passthrough']) | qa_prompt | llm | StrOutputParser()

# --- Chain 4: Dividend Question ---

# FIX: This chain was failing because the parser was receiving the wrong data structure.
# The new structure explicitly passes the correct 'input' to the parser and correctly
# formats the final output string, including converting the yield ratio to a percentage.
dividend_chain = (
    {
        # Explicitly pass the input string to the parser.
        "ticker": (lambda x: parse_ticker_from_input({'input': x['passthrough']['input']})),
        "passthrough": (lambda x: x['passthrough'])
    }
    | RunnableLambda(lambda x: {
        "ticker": x['ticker'],
        "info": get_dividend_info.invoke(x['ticker'])
    })
    | RunnableLambda(lambda x: {
        "ticker": x['ticker'],
        "yield_percent": (x['info'].get('dividend_yield_percent') * 100) if x['info'].get('dividend_yield_percent') is not None else 'N/A',
        "payout": x['info'].get('last_payout_idr', 'N/A')
    })
    | (lambda x: f"The dividend yield for {x['ticker']} is {x['yield_percent']:.2f}% with a last payout of Rp {x['payout']:.2f}." if isinstance(x['yield_percent'], float) else f"The dividend yield for {x['ticker']} is {x['yield_percent']}% with a last payout of Rp {x['payout']}.")
)

# --- 4. BUILD THE MAIN ROUTER CHAIN ---

router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

# The router itself, which outputs a classification string
router = router_prompt | llm | StrOutputParser()

branch = RunnableBranch(
    (lambda x: "comparison" in x["intent"].lower(), stock_comparison_chain),
    (lambda x: "analysis" in x["intent"].lower(), stock_analysis_chain),
    (lambda x: "dividend" in x["intent"].lower(), dividend_chain),
    general_conversation_chain,  # Default case
)

# The full chain that includes the router and the branch
full_chain = RunnableParallel(
    # FIX: The 'intent' key was missing. This restores the router to the chain.
    # The router runs in parallel to the passthrough.
    intent=router,
    passthrough=RunnablePassthrough()
) | RunnableLambda(
    lambda x: branch.invoke(x)
)


# This is the final chain that will be called by the app.
# We name it 'agent_executor' to avoid having to change the app.py file.
agent_executor = full_chain