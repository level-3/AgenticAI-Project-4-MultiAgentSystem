from enum import Enum
import logging
import re
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import uuid
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Any, Dict, List, TypedDict, Union
from sqlalchemy import create_engine, Engine
from decimal import Decimal
from smolagents import CodeAgent, ToolCallingAgent, tool, OpenAIServerModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class OrderSize(str, Enum):
    """Order size categories for discount calculation."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")


def normalize_agent_response(response: Any) -> Dict[str, Any]:
    """
    Normalize agent response to always return a dictionary.

    Handles cases where agents return:
    - Already a dict → return as-is
    - JSON string → parse and return dict
    - Escaped JSON with literal newlines → fix and parse
    - AgentText type → convert to string then parse
    - Other formats → try to extract JSON from string

    Args:
        response: Agent response in any format

    Returns:
        Dictionary representation of the response

    Raises:
        ValueError: If response cannot be parsed into a dictionary
    """
    # Already a dictionary
    if isinstance(response, dict):
        return response

    # Handle smolagents AgentText type
    if hasattr(response, "__class__") and "AgentText" in response.__class__.__name__:
        response = str(response)

    # String that might be JSON
    if isinstance(response, str):
        response = response.strip()

        # Try direct JSON parse first (standard JSON)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Handle cases where \n appears as literal string characters (not escaped)
        # This happens when agents return strings like: '{\n  "key": "value"\n}'
        if '\\n' in response:
            try:
                # Replace literal \n with actual newlines
                cleaned = response.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Handle escaped quotes with literal newlines (common with smolagents)
        # Example: {\"key\": \"value\nwith newline\"}
        if '\\"' in response:
            try:
                # Step 1: Replace escaped quotes with regular quotes
                unescaped = response.replace('\\"', '"')

                # Step 2: Escape any literal control characters
                # These are actual newlines/tabs in the string that break JSON parsing
                unescaped = unescaped.replace("\n", "\\n")
                unescaped = unescaped.replace("\r", "\\r")
                unescaped = unescaped.replace("\t", "\\t")

                # Step 3: Try parsing
                parsed = json.loads(unescaped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Handle literal newlines without escaped quotes
        # Sometimes JSON has newlines but no escaped quotes
        if "\n" in response or "\r" in response or "\t" in response:
            try:
                # Escape control characters
                fixed = response.replace("\n", "\\n")
                fixed = fixed.replace("\r", "\\r")
                fixed = fixed.replace("\t", "\\t")

                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Handle double-escaped JSON (wrapped in quotes)
        if response.startswith('"') and response.endswith('"'):
            try:
                # Parse once to remove outer quotes
                unquoted = json.loads(response)
                if isinstance(unquoted, str):
                    # Try parsing again
                    parsed = json.loads(unquoted)
                    if isinstance(parsed, dict):
                        return parsed
                elif isinstance(unquoted, dict):
                    return unquoted
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            try:
                parsed = json.loads(matches[0])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                # Try with control character escaping
                try:
                    fixed = (
                        matches[0]
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                    )
                    parsed = json.loads(fixed)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        # Try to find JSON object in the string
        # Look for outermost curly braces
        start_idx = response.find("{")
        if start_idx != -1:
            # Find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(response)):
                if response[i] == "{":
                    brace_count += 1
                elif response[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_idx : i + 1]

                        # Try parsing as-is
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and len(parsed) > 0:
                                return parsed
                        except json.JSONDecodeError:
                            pass

                        # Try with escaped quotes replaced
                        try:
                            unescaped = json_str.replace('\\"', '"')
                            # Also escape control characters
                            unescaped = (
                                unescaped.replace("\n", "\\n")
                                .replace("\r", "\\r")
                                .replace("\t", "\\t")
                            )
                            parsed = json.loads(unescaped)
                            if isinstance(parsed, dict) and len(parsed) > 0:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

        # Try ast.literal_eval for Python dict literals (single quotes)
        # Replace JSON literals with Python equivalents
        response_py = (
            response.replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        try:
            parsed = ast.literal_eval(response_py)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # If all parsing fails, raise an error with context
        raise ValueError(
            f"Could not parse agent response into dictionary. "
            f"Response type: {type(response)}, "
            f"First 200 chars: {str(response)[:200]}"
        )

    # Other types
    raise TypeError(f"Agent response must be dict or JSON string, got {type(response)}")


JSON_RESPONSE_INSTRUCTIONS = """
CRITICAL JSON FORMATTING RULES:

When returning JSON responses, you MUST follow these rules exactly:

1. ALWAYS return valid, parseable JSON
2. Use DOUBLE QUOTES for all strings (not single quotes)
3. Do NOT escape quotes in your response
4. For multi-line text, use the escaped newline character \\n (not actual line breaks)
5. Return ONLY the JSON object - no markdown, no code blocks, no explanations
6. Ensure all brackets and braces are properly closed
7. Use null for missing values (not None, not undefined)
8. Use true/false for booleans (not True/False)

CORRECT FORMAT:
{
  "key": "value",
  "multiline": "Line 1\\nLine 2\\nLine 3",
  "number": 42,
  "boolean": true,
  "nullable": null,
  "list": ["item1", "item2"]
}

INCORRECT FORMATS (DO NOT USE):
❌ {\"key\": \"value\"}  (escaped quotes)
❌ {"key": "Line 1
    Line 2"}  (literal newlines)
❌ {'key': 'value'}  (single quotes)
❌ ```json {...}```  (markdown code blocks)
❌ {"key": None}  (Python None instead of null)

VALIDATION CHECKLIST:
- [ ] All strings use double quotes
- [ ] No escaped quotes in the outer structure
- [ ] Multi-line strings use \\n not actual newlines
- [ ] All JSON keys are strings
- [ ] Proper comma placement
- [ ] Valid JSON syntax throughout

Your response will be parsed directly with json.loads() - ensure it's valid!
"""


# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},
    # Product Types (priced per unit)
    {
        "item_name": "Paper plates",
        "category": "product",
        "unit_price": 0.10,
    },  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {
        "item_name": "Paper napkins",
        "category": "product",
        "unit_price": 0.02,
    },  # per napkin
    {
        "item_name": "Disposable cups",
        "category": "product",
        "unit_price": 0.10,
    },  # per cup
    {
        "item_name": "Table covers",
        "category": "product",
        "unit_price": 1.50,
    },  # per cover
    {
        "item_name": "Envelopes",
        "category": "product",
        "unit_price": 0.05,
    },  # per envelope
    {
        "item_name": "Sticky notes",
        "category": "product",
        "unit_price": 0.03,
    },  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {
        "item_name": "Invitation cards",
        "category": "product",
        "unit_price": 0.50,
    },  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {
        "item_name": "Party streamers",
        "category": "product",
        "unit_price": 0.05,
    },  # per roll
    {
        "item_name": "Decorative adhesive tape (washi tape)",
        "category": "product",
        "unit_price": 0.20,
    },  # per roll
    {
        "item_name": "Paper party bags",
        "category": "product",
        "unit_price": 0.25,
    },  # per bag
    {
        "item_name": "Name tags with lanyards",
        "category": "product",
        "unit_price": 0.75,
    },  # per tag
    {
        "item_name": "Presentation folders",
        "category": "product",
        "unit_price": 0.50,
    },  # per folder
    # Large-format items (priced per unit)
    {
        "item_name": "Large poster paper (24x36 inches)",
        "category": "large_format",
        "unit_price": 1.00,
    },
    {
        "item_name": "Rolls of banner paper (36-inch width)",
        "category": "large_format",
        "unit_price": 2.50,
    },
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

logger = logging.getLogger(__name__)

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
            - item_name
            - category
            - unit_price
            - current_stock
            - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_min_stock_level(item_name: str) -> pd.DataFrame:
    """
    Retrieve the minimum stock level for a specific item.

    Args:
        item_name (str): The name of the item to look up.

    Returns:
        int: The minimum stock level for the item.
    """
    # SQL query to retrieve the minimum stock level for the item
    stock_query = """
        SELECT min_stock_level
        FROM (
            SELECT
                item_name,min_stock_level
                FROM inventory
                WHERE item_name = :item_name
        )
    """

    # Execute query and return result as a DataFrame    \
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base."
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> str:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        str: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " OR ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        result_list = result.fetchall()

        """
        print(f"Executed quote history search with terms: {search_terms}")
        print(f"SQL Query: {query}")
        print(f"Query Parameters: {params}")
        print(f"Number of results: {result.rowcount}")
        print(f"Results: {result_list}")
        """

        return result_list
        # return [dict(row) for row in result]


def remove_quotes(text: str) -> str:
    """
    Remove all double quote characters from a string.

    This function removes all occurrences of the double quote character (")
    from the input string. Useful for sanitizing user input, cleaning data
    from external sources, or preparing strings for database operations.

    Args:
        text: The string to clean. Can be empty or contain any characters.

    Returns:
        A new string with all double quotes removed. Returns empty string
        if input is empty.

    Examples:
        >>> remove_quotes('Hello "World"')
        'Hello World'

        >>> remove_quotes('"Quoted text"')
        'Quoted text'

        >>> remove_quotes('No quotes here')
        'No quotes here'

        >>> remove_quotes('Multiple ""quotes"" here')
        'Multiple quotes here'

        >>> remove_quotes('')
        ''
    """
    return text.replace('"', "")


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
# Tools for quoting agent
# Tools for ordering agent
# Set up your agents and create an orchestration agent that will manage them.
# Run your test scenarios by writing them here. Make sure to keep track of them.

# Load environment variables from Vocareum config
dotenv.load_dotenv("/home/level-3/udacity/AgenticAI/.env/config.env")
openai_api_key = os.getenv("VOCAREUM_API_KEY")


def get_llm_model():
    """Initialize LLM model for Vocareum environment."""
    if not openai_api_key:
        raise ValueError("VOCAREUM_API_KEY not found in environment variables")

    return OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_base="https://openai.vocareum.com/v1",
        api_key=openai_api_key,
    )


# Tools


class InventorySemanticSearch:
    """
    Semantic search engine for inventory items and product catalog using TF-IDF.

    Provides fast keyword-based semantic search without requiring external APIs
    or heavy machine learning dependencies.

    Features:
    - Search current inventory (database items with stock levels)
    - Search product catalog (all available products from paper_supplies)
    - TF-IDF vectorization for efficient similarity matching
    """

    def __init__(self, db_engine, paper_supplies: list[dict] | None = None):
        """
        Initialize semantic search engine with TF-IDF.

        Args:
            db_engine: SQLAlchemy engine for database access
            paper_supplies: List of product dictionaries from catalog
                          Each dict should have: item_name, category, unit_price
        """
        self.db_engine = db_engine
        self.paper_supplies = paper_supplies or []

        # Inventory search attributes (database items)
        self.vectorizer: TfidfVectorizer | None = None
        self.item_vectors: np.ndarray | None = None
        self.items_df: pd.DataFrame | None = None

        # Catalog search attributes (paper_supplies)
        self.catalog_vectorizer: TfidfVectorizer | None = None
        self.catalog_vectors: np.ndarray | None = None
        self.catalog_df: pd.DataFrame | None = None

        print("🔍 Initializing semantic search with TF-IDF")

        # Load inventory and create embeddings
        self._initialize_embeddings()

        # Initialize catalog embeddings if paper_supplies provided
        if self.paper_supplies:
            self._initialize_catalog_embeddings()

    def _initialize_embeddings(self) -> None:
        """Load inventory from database and create TF-IDF vector embeddings."""
        # Load all inventory items from database
        query = "SELECT item_name, category, unit_price FROM inventory"
        self.items_df = pd.read_sql(query, self.db_engine)

        if self.items_df.empty:
            print("⚠️  No items in inventory to vectorize")
            return

        # Create searchable text for each item (combine name and category)
        self.items_df["search_text"] = self.items_df.apply(
            lambda row: f"{row['item_name']} {row['category']}", axis=1
        )

        # Create TF-IDF embeddings for inventory
        self._create_tfidf_embeddings()

        print(f"✅ Vectorized {len(self.items_df)} inventory items using TF-IDF")

    def _initialize_catalog_embeddings(self) -> None:
        """Load paper_supplies catalog and create TF-IDF vector embeddings."""
        if not self.paper_supplies:
            print("⚠️  No catalog items to vectorize")
            return

        # Convert paper_supplies list to DataFrame
        self.catalog_df = pd.DataFrame(self.paper_supplies)

        if self.catalog_df.empty:
            print("⚠️  Empty catalog provided")
            return

        # Create searchable text for each catalog item
        self.catalog_df["search_text"] = self.catalog_df.apply(
            lambda row: f"{row['item_name']} {row['category']}", axis=1
        )

        # Create TF-IDF embeddings for catalog
        self._create_catalog_tfidf_embeddings()

        print(f"✅ Vectorized {len(self.catalog_df)} catalog items using TF-IDF")

    def _create_tfidf_embeddings(self) -> None:
        """Create embeddings for inventory using TF-IDF vectorization."""
        texts = self.items_df["search_text"].tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        self.item_vectors = self.vectorizer.fit_transform(texts).toarray()

    def _create_catalog_tfidf_embeddings(self) -> None:
        """Create embeddings for catalog using TF-IDF vectorization."""
        texts = self.catalog_df["search_text"].tolist()

        self.catalog_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        self.catalog_vectors = self.catalog_vectorizer.fit_transform(texts).toarray()

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get TF-IDF embedding vector for a search query (inventory).

        Args:
            query: Search query string

        Returns:
            Numpy array of TF-IDF features

        Raises:
            RuntimeError: If vectorizer not initialized
        """
        if self.vectorizer is None:
            raise RuntimeError("Inventory vectorizer not initialized")

        return self.vectorizer.transform([query]).toarray()[0]

    def _get_catalog_query_embedding(self, query: str) -> np.ndarray:
        """
        Get TF-IDF embedding vector for a catalog search query.

        Args:
            query: Search query string

        Returns:
            Numpy array of TF-IDF features

        Raises:
            RuntimeError: If catalog vectorizer not initialized
        """
        if self.catalog_vectorizer is None:
            raise RuntimeError("Catalog vectorizer not initialized")

        return self.catalog_vectorizer.transform([query]).toarray()[0]

    def search(
        self, query: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> list[dict]:
        """
        Search inventory (database items) using TF-IDF semantic similarity.

        Args:
            query: Search query (natural language or keywords)
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold (0.0 to 1.0)

        Returns:
            List of matching items with similarity scores
        """
        if self.items_df is None or self.items_df.empty:
            return []

        if self.item_vectors is None:
            return []

        # Get query embedding
        query_vector = self._get_query_embedding(query)

        # Reshape for cosine similarity calculation
        query_vector = query_vector.reshape(1, -1)

        # Calculate cosine similarities between query and all items
        similarities = cosine_similarity(query_vector, self.item_vectors)[0]

        # Get top-k indices (sorted by similarity, descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list, filtering by minimum similarity
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])

            if similarity < min_similarity:
                continue

            item = self.items_df.iloc[idx]
            results.append(
                {
                    "item_name": item["item_name"],
                    "category": item["category"],
                    "unit_price": float(item["unit_price"]),
                    "similarity_score": similarity,
                    "confidence": self._get_confidence_level(similarity),
                    "source": "inventory",
                }
            )

        return results

    def search_catalog(
        self, query: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> list[dict]:
        """
        Search product catalog (paper_supplies) using TF-IDF semantic similarity.

        Args:
            query: Search query (natural language or keywords)
            top_k: Number of top results to return
            min_similarity: Minimum cosine similarity threshold (0.0 to 1.0)

        Returns:
            List of matching catalog items with similarity scores
        """
        if self.catalog_df is None or self.catalog_df.empty:
            return []

        if self.catalog_vectors is None:
            return []

        # Get query embedding using catalog vectorizer
        query_vector = self._get_catalog_query_embedding(query)

        # Reshape for cosine similarity calculation
        query_vector = query_vector.reshape(1, -1)

        # Calculate cosine similarities between query and all catalog items
        similarities = cosine_similarity(query_vector, self.catalog_vectors)[0]

        # Get top-k indices (sorted by similarity, descending)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results list, filtering by minimum similarity
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])

            if similarity < min_similarity:
                continue

            item = self.catalog_df.iloc[idx]
            results.append(
                {
                    "item_name": item["item_name"],
                    "category": item["category"],
                    "unit_price": float(item["unit_price"]),
                    "similarity_score": similarity,
                    "confidence": self._get_confidence_level(similarity),
                    "source": "catalog",
                }
            )

        return results

    def _get_confidence_level(self, similarity: float) -> str:
        """
        Convert similarity score to human-readable confidence level.

        Args:
            similarity: Cosine similarity score (0.0 to 1.0)

        Returns:
            Confidence level: "high", "medium", "low", or "very_low"
        """
        if similarity >= 0.8:
            return "high"
        elif similarity >= 0.6:
            return "medium"
        elif similarity >= 0.4:
            return "low"
        else:
            return "very_low"

    def find_best_match(self, query: str, threshold: float = 0.5) -> dict | None:
        """
        Find the single best matching item from inventory above a threshold.

        Args:
            query: Search query
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            Best matching inventory item or None if no good match found
        """
        results = self.search(query, top_k=1, min_similarity=threshold)
        return results[0] if results else None

    def find_best_catalog_match(
        self, query: str, threshold: float = 0.4
    ) -> dict | None:
        """
        Find the single best matching item from catalog above a threshold.

        Args:
            query: Search query
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            Best matching catalog item or None if no good match found
        """
        results = self.search_catalog(query, top_k=1, min_similarity=threshold)
        return results[0] if results else None

    def find_multiple_catalog_matches(
        self, query: str, threshold: float = 0.4
    ) -> dict | None:
        """
        Find the best matching items from catalog above a threshold.

        Args:
            query: Search query
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            Best matching catalog item or None if no good match found
        """
        results = self.search_catalog(query, top_k=3, min_similarity=threshold)
        return results if results else None

    def search_both(
        self, query: str, top_k: int = 5, min_similarity: float = 0.0
    ) -> dict[str, list[dict]]:
        """
        Search both inventory and catalog simultaneously.

        Useful for finding items in stock and available alternatives from catalog.

        Args:
            query: Search query (natural language or keywords)
            top_k: Number of top results to return from each source
            min_similarity: Minimum cosine similarity threshold (0.0 to 1.0)

        Returns:
            Dictionary with 'inventory' and 'catalog' keys containing results
        """
        inventory_results = self.search(query, top_k, min_similarity)
        catalog_results = self.search_catalog(query, top_k, min_similarity)
        # print("🔍 Searching both inventory and catalog for:", query)
        # print("🔍 Inventory search results:", inventory_results)
        # print("🔍 Catalog search results:", catalog_results)

        return {
            "inventory": inventory_results,
            "catalog": catalog_results,
            "query": query,
        }

    def find_catalog_items_not_in_inventory(self) -> list[dict]:
        """
        Find items that exist in catalog but not in current inventory.

        Useful for identifying potential new products to stock.

        Returns:
            List of catalog items not currently in inventory
        """
        if self.catalog_df is None or self.items_df is None:
            return []

        # Get item names from inventory
        inventory_items = set(self.items_df["item_name"].tolist())

        # Get catalog items not in inventory
        not_in_inventory = self.catalog_df[
            ~self.catalog_df["item_name"].isin(inventory_items)
        ]

        return not_in_inventory[["item_name", "category", "unit_price"]].to_dict(
            "records"
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the search engine.

        Returns:
            Dictionary with inventory and catalog statistics
        """
        inventory_count = len(self.items_df) if self.items_df is not None else 0
        catalog_count = len(self.catalog_df) if self.catalog_df is not None else 0

        return {
            "inventory_items": inventory_count,
            "catalog_items": catalog_count,
            "items_not_in_inventory": len(self.find_catalog_items_not_in_inventory()),
            "inventory_vectorized": self.item_vectors is not None,
            "catalog_vectorized": self.catalog_vectors is not None,
        }


def initialize_semantic_search(
    db_engine, paper_supplies: list[dict] | None = None
) -> InventorySemanticSearch:
    """
    Initialize the global semantic search instance.

    Args:
        db_engine: SQLAlchemy database engine
        paper_supplies: List of product dictionaries for catalog search

    Returns:
        Initialized InventorySemanticSearch instance
    """
    global _semantic_search_instance
    _semantic_search_instance = InventorySemanticSearch(db_engine, paper_supplies)
    return _semantic_search_instance


def get_semantic_search() -> InventorySemanticSearch | None:
    """
    Get the global semantic search instance.

    Returns:
        InventorySemanticSearch instance or None if not initialized
    """
    return _semantic_search_instance


# Example usage with tools
def semantic_search_inventory(query: str, request_date: str) -> dict[str, Any]:
    """
    Perform semantic search on inventory and return JSON results.

    Args:
        query: Search term keywords
        request_date: Date to check stock as of (ISO format YYYY-MM-DD)

    Returns:
        Dictionary with search results
    """
    search_engine = get_semantic_search()

    if not search_engine:
        return {
            "error": "Semantic search not initialized",
            "message": "Call initialize_semantic_search() first",
        }

    # print(f"🔍 Searching inventory for: '{query}'")

    results = search_engine.find_best_match(query, threshold=0.5)

    if not results:
        return {
            "found": False,
            "message": f"No similar items found in inventory for '{query}'",
            "results": [],
        }

    return (
        {
            "item_name": results["item_name"],
            "category": results["category"],
            "unit_price": f"${results['unit_price']:.2f}",
            "similarity_score": results["similarity_score"],
            "confidence": results["confidence"],
            "found": True,
            "query": query,
            "source": "inventory",
        },
    )


def _check_exact_match(query: str, catalog_df: pd.DataFrame) -> dict | None:
    """
    Check if query exactly matches any item name (case-insensitive).

    Args:
        query: Search query string
        catalog_df: DataFrame containing catalog items

    Returns:
        Dictionary with item details if exact match found, None otherwise
    """
    if catalog_df is None or catalog_df.empty:
        return None

    query_lower = query.lower().strip()
    matches = catalog_df[catalog_df["item_name"].str.lower() == query_lower]

    if not matches.empty:
        return matches.iloc[0].to_dict()
    return None


def _validate_category_match(query: str, matched_category: str) -> bool:
    """
    Validate if the matched item's category makes sense for the query.

    Returns False if the match is semantically nonsensical.

    Args:
        query: Original search query
        matched_category: Category of the matched item

    Returns:
        True if match is valid, False if semantically incorrect
    """
    # Define categories that don't make sense for specific queries
    invalid_matches = {
        "tickets": [
            "paper",
            "specialty",
            "large_format",
        ],  # tickets shouldn't match paper types
        "balloons": [
            "paper",
            "specialty",
            "product",
            "large_format",
        ],  # balloons aren't paper products
        "decorations": [
            "paper",
            "specialty",
        ],  # general decorations shouldn't match paper
        "banners": ["paper"],  # finished banners vs banner paper (raw material)
    }

    query_lower = query.lower().strip()
    if query_lower in invalid_matches:
        return matched_category not in invalid_matches[query_lower]

    return True  # Allow match by default


@tool
def semantic_search_catalog(query: str) -> dict[str, Any]:
    """
    Perform semantic search on product catalog and return JSON results.

    Args:
        query (str): Search term keywords

    Returns:
        Dictionary with search results

        Example:
        {
            "item_name": "Premium Copy Paper",
            "category": "Paper",
            "unit_price": "$12.99",
            "similarity_score": 0.85,
            "confidence": 0.95,
            "found": True,
            "query": "copy paper",
            "source": "catalog",
            "message": "Found similar item in catalog, item can be ordered"
        }
    """
    search_engine = get_semantic_search()

    if not search_engine:
        return {
            "error": "Semantic search not initialized",
            "message": "Call initialize_semantic_search() first",
        }

    print(f"🔍 Searching catalog for: '{query}'")

    # ✅ FIX 1: Check for exact match first
    exact_match = _check_exact_match(query, search_engine.catalog_df)
    if exact_match:
        print(f"✅ Exact match found for '{query}': {exact_match['item_name']}")
        return [
            {
                "item_name": exact_match["item_name"],
                "category": exact_match["category"],
                "unit_price": f"${exact_match['unit_price']:.2f}",
                "confidence": "high",
                "found": True,
                "query": query,
                "source": "catalog",
                "message": "Exact match found in catalog, item can be ordered",
            }
        ]

    # ✅ FIX 2: Increase threshold from 0.4 to 0.65 for stricter matching
    results = search_engine.find_multiple_catalog_matches(query, threshold=0.65)

    # ✅ FIX 3: Log all matches for debugging
    if results:
        print(f"📊 Semantic search results for '{query}':")
        for r in results:
            print(
                f"   - {r['item_name']} (confidence: {r['confidence']}, similarity: {r['similarity_score']:.3f})"
            )
    else:
        print(f"❌ No matches found for '{query}' (threshold: 0.65)")

    if not results:
        return {
            "found": False,
            "message": f"Similar item NOT found in catalog, item cannot be ordered",
            "results": [],
        }

    # ✅ FIX 4: Filter results by confidence level (only high/medium)
    high_confidence_results = [
        r for r in results if r["confidence"] in ["high", "medium"]
    ]

    if not high_confidence_results:
        print(
            f"⚠️  No high-confidence matches for '{query}' (all matches were low confidence)"
        )
        return {
            "found": False,
            "message": f"No high-confidence matches found for '{query}'",
            "results": [],
        }

    # ✅ FIX 5: Validate category makes sense for the query
    validated_results = []
    for result in high_confidence_results:
        if _validate_category_match(query, result["category"]):
            validated_results.append(result)
        else:
            print(
                f"⚠️  Rejected match '{result['item_name']}' for query '{query}' - category mismatch"
            )

    if not validated_results:
        print(
            f"⚠️  No semantically valid matches for '{query}' (all matches failed category validation)"
        )
        return {
            "found": False,
            "message": f"No semantically appropriate matches found for '{query}'",
            "results": [],
        }

    item_list = []
    for result in validated_results:
        item = {
            "item_name": result["item_name"],
            "category": result["category"],
            "unit_price": f"${result['unit_price']:.2f}",
            "confidence": result["confidence"],
            "found": True,
            "query": query,
            "source": "catalog",
            "message": f"Found similar item in catalog, item can be ordered",
        }
        item_list.append(item)

    print(f"✅ Returning {len(item_list)} validated match(es) for '{query}'")
    return item_list


def semantic_search_both(query: str) -> dict[str, Any]:
    """
    Search both inventory and catalog simultaneously.

    Args:
        query: Search term keyword maximum 1 word
        top_k: Number of results from each source

    Returns:
        Dictionary with results from both sources
    """
    search_engine = get_semantic_search()

    if not search_engine:
        return {
            "error": "Semantic search not initialized",
            "message": "Call initialize_semantic_search() first",
        }

    results = search_engine.search_catalog(query, top_k=1, min_similarity=0.4)

    if len(results) > 0:
        return (
            {
                "item_name": results["item_name"],
                "category": results["category"],
                "unit_price": f"${results['unit_price']:.2f}",
                "similarity_score": results["similarity_score"],
                "confidence": results["confidence"],
                "found": True,
                "query": query,
                "source": "catalog",
            },
        )

    elif len(results) == 0:
        return (
            {
                "found": False,
                "message": f"No similar items found in catalog for '{query}'",
                "query": query,
            },
        )


@tool
def calculate_bulk_discount(total_quantity: int, need_size: str) -> float:
    """
    Calculate bulk discount.

    Args:
        total_quantity (int): Total quantity of items in the order.
        need_size (str): Size of the order (SMALL, MEDIUM, LARGE).

    Returns:
        float: Total discount percentage.
    """
    if total_quantity >= 10000:
        base_discount = 10.0
    elif total_quantity >= 5000:
        base_discount = 5.0
    elif total_quantity >= 1000:
        base_discount = 3.0
    else:
        base_discount = 0.0

    size_bonus = 2.0 if need_size == OrderSize.LARGE else 0.0
    return base_discount + size_bonus


def check_reorder_needs() -> str:
    """Check items needing reorder.

    Returns:
        str: dictionary with reorder recommendations.
    """
    try:
        query = "SELECT * FROM inventory WHERE current_stock < min_stock_level ORDER BY (current_stock - min_stock_level)"
        low_stock_items = pd.read_sql(query, db_engine)

        if low_stock_items.empty:
            return json.dumps(
                {
                    "items_needing_reorder": 0,
                    "items": [],
                    "total_estimated_cost": 0.0,
                    "message": "All items are adequately stocked",
                },
                indent=2,
            )

        reorder_list = []
        for _, item in low_stock_items.iterrows():
            shortage = int(item["min_stock_level"] - item["current_stock"])
            recommended_order = max(shortage, int(item["min_stock_level"]) * 2)
            reorder_list.append(
                {
                    "item_name": item["item_name"],
                    "category": item["category"],
                    "current_stock": int(item["current_stock"]),
                    "min_stock_level": int(item["min_stock_level"]),
                    "shortage": shortage,
                    "recommended_order_quantity": recommended_order,
                    "estimated_cost": float(recommended_order * item["unit_price"]),
                }
            )

        total_cost = sum(item["estimated_cost"] for item in reorder_list)

        return json.dumps(
            {
                "items_needing_reorder": len(reorder_list),
                "items": reorder_list,
                "total_estimated_cost": total_cost,
                "message": f"{len(reorder_list)} item(s) need reordering - estimated cost: ${total_cost:.2f}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool
def check_current_stock(item_name: str, request_date: str, found: bool) -> str:
    """Check current stock level for an item.

    Args:
        item_name: Name of the item, cannot be "Not found" or "None".
        request_date: Date of the request in ISO format.
        found: Boolean indicating if the item was found in catalog.

    Returns:
        dictionary with current stock information.
    """

    if found == False:
        return json.dumps(
            {
                "item_name": item_name,
                "message": f"Item cannot be ordered",
            },
            indent=2,
        )

    try:
        # Handle items that cannot be ordered
        if item_name == "Not found" or item_name == "None":
            return json.dumps(
                {
                    "item_name": item_name,
                    "message": "Item cannot be ordered",
                },
                indent=2,
            )

        # Get stock information
        stock_info = get_stock_level(item_name, request_date)
        # print(f"stock_info: \n{stock_info}")

        # Handle empty stock info
        if stock_info.empty:
            print(f"stock_info is empty for {item_name}")
            return {
                "item_name": item_name,
                "current_stock": 0,
                "min_stock_level": 0,
                "request_date": request_date,
                "message": f"Current stock for '{item_name}' as of {request_date} is 0 unit(s).",
            }

        # Stock info exists - extract current stock
        # print(f"stock_info is not empty for {item_name}")
        current_stock = int(stock_info["current_stock"].iloc[0])

        # Get minimum stock level
        min_stock_level_info = get_min_stock_level(item_name)

        # If DataFrame is NOT empty, access the value
        if not min_stock_level_info.empty:  # Fixed condition
            min_stock_level = int(min_stock_level_info["min_stock_level"].iloc[0])
        else:
            min_stock_level = 0

        return {
            "item_name": item_name,
            "current_stock": current_stock,
            "min_stock_level": min_stock_level,
            "request_date": request_date,
            "message": f"Current stock for '{item_name}' as of {request_date} is {current_stock} unit(s).",
        }

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool
def update_inventory(
    item_name: str,
    transaction_type: str,
    order_quantity: int,
    price: float,
    request_date: str,
) -> str:
    """Update inventory with new quantity.

    Args:
        item_name (str): Name of the item.
        transaction_type (str): Type of transaction, either 'stock_orders' or 'sales'.
        order_quantity (int): Quantity of order.
        price (float): Total value of the transaction.
        request_date (str): Date of the request in ISO format.


    Returns:
        str: dictionary with updated inventory information.
    """
    try:
        # create a transaction to update inventory
        transaction_id = create_transaction(
            item_name=item_name,
            transaction_type=transaction_type,
            quantity=order_quantity,
            price=price,
            date=request_date,
        )

        return json.dumps(
            {
                "item_name": item_name,
                "quantity": order_quantity,
                "transaction_type": transaction_type,
                "transaction_id": transaction_id,
                "message": f"Inventory updated for '{item_name}' with quantity {order_quantity}. Transaction ID: {transaction_id}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


def reorder_item(
    item_name: str,
    reorder_quantity: int,
    request_date: str,
    unit_price: float,
    category: str = "paper",
    min_stock_level: int | None = None,
) -> str:
    """
    Reorder item from supplier. Adds item to inventory if not found.

    This function:
    1. Checks if the item exists in the inventory table
    2. If not found, creates a new inventory record with the reorder quantity as initial stock
    3. If found, updates the current stock by adding the reorder quantity
    4. Records the transaction in the transactions table

    Args:
        item_name: Name of the item to reorder
        reorder_quantity: Quantity to reorder (must be > 0)
        request_date: Date of the request in ISO format (YYYY-MM-DD)
        unit_price: Unit price of the item (must be >= 0)
        category: Category for new items (default: "paper").
                 Options: "paper", "specialty", "large_format", "product"
        min_stock_level: Minimum stock level for new items.
                        Defaults to 20% of reorder_quantity if not specified

    Returns:
        JSON string with reorder information including whether item was newly added

    Examples:
        reorder_item("Cardstock", 1000, "2025-01-15", 0.15)
        reorder_item("Custom Paper", 500, "2025-01-15", 0.25, category="specialty")
    """
    try:
        print(f"Processing reorder: {item_name}, quantity: {reorder_quantity}")

        # Validate inputs
        if not item_name or not item_name.strip():
            return json.dumps(
                {"success": False, "error": "item_name cannot be empty"}, indent=2
            )

        if reorder_quantity <= 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"reorder_quantity must be positive, got {reorder_quantity}",
                },
                indent=2,
            )

        if unit_price < 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"unit_price must be non-negative, got {unit_price}",
                },
                indent=2,
            )

        # Validate category
        valid_categories = ["paper", "specialty", "large_format", "product"]
        if category not in valid_categories:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}",
                },
                indent=2,
            )

        # Calculate default min_stock_level if not provided
        if min_stock_level is None:
            min_stock_level = min(50, int(reorder_quantity * 0.2))

        # Check if item exists in inventory
        check_query = (
            f"SELECT * FROM inventory WHERE LOWER(item_name) = LOWER('{item_name}')"
        )
        existing_items = pd.read_sql(check_query, db_engine)

        item_was_new = False
        previous_stock = 0

        if existing_items.empty:
            # Item not found - add to inventory
            print(f"Item '{item_name}' not found in inventory. Adding new item.")

            insert_query = f"""
                INSERT INTO inventory (item_name, category, unit_price, current_stock, min_stock_level)
                VALUES ('{item_name}', '{category}', {unit_price}, {reorder_quantity}, {min_stock_level})
            """

            with db_engine.connect() as conn:
                conn.execute(text(insert_query))
                conn.commit()

            print(
                f"New item added to inventory: {item_name} "
                f"(category: {category}, stock: {reorder_quantity}, min: {min_stock_level})"
            )

            item_was_new = True
            new_stock = reorder_quantity

        else:
            # Item exists - update stock
            item = existing_items.iloc[0].to_dict()
            previous_stock = int(item["current_stock"])
            new_stock = previous_stock + reorder_quantity

            print(
                f"Item '{item_name}' found in inventory. "
                f"Updating stock: {previous_stock} -> {new_stock}"
            )

            update_query = f"""
                UPDATE inventory 
                SET current_stock = {new_stock}
                WHERE LOWER(item_name) = LOWER('{item_name}')
            """

            with db_engine.connect() as conn:
                conn.execute(text(update_query))
                conn.commit()

            print(
                f"Inventory updated for {item_name}: stock increased by {reorder_quantity}"
            )

        # Create transaction record
        total_cost = reorder_quantity * unit_price
        order_id = create_transaction(
            item_name=item_name,
            price=total_cost,
            transaction_type="stock_orders",
            quantity=reorder_quantity,
            date=request_date,
        )

        print(
            f"Transaction recorded: Order ID {order_id}, "
            f"{reorder_quantity} units @ ${unit_price:.2f} = ${total_cost:.2f}"
        )

        # Get updated inventory status
        updated_query = (
            f"SELECT * FROM inventory WHERE LOWER(item_name) = LOWER('{item_name}')"
        )
        updated_item = pd.read_sql(updated_query, db_engine).iloc[0].to_dict()

        # Check stock status
        current_stock = updated_item["current_stock"]
        min_level = updated_item["min_stock_level"]

        if current_stock >= min_level:
            stock_status = (
                f"✓ Stock healthy: {current_stock} units (above minimum of {min_level})"
            )
        else:
            stock_status = (
                f"⚠️ Still below minimum: {current_stock} units (minimum: {min_level})"
            )

        # Build result
        result = {
            "success": True,
            "item_name": item_name,
            "category": updated_item["category"],
            "reorder_quantity": reorder_quantity,
            "unit_price": float(unit_price),
            "total_cost": float(total_cost),
            "reorder_date": request_date,
            "order_id": order_id,
            "inventory_status": {
                "was_new_item": item_was_new,
                "previous_stock": previous_stock,
                "new_stock": current_stock,
                "min_stock_level": min_level,
                "stock_status": stock_status,
            },
            "message": (
                f"✓ Reorder placed for '{item_name}': {reorder_quantity} units @ ${unit_price:.2f} = ${total_cost:.2f}. "
                f"Order ID: {order_id}. "
                f"{'[NEW ITEM ADDED TO INVENTORY]' if item_was_new else f'Stock: {previous_stock} → {current_stock}'}"
            ),
        }

        print(f"Reorder completed successfully: {result['message']}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Error processing reorder for {item_name}: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def reorder_item_tool(
    item_name: str,
    reorder_quantity: int,
    request_date: str,
    unit_price: float,
    category: str,
    min_stock_level: int = 0,
) -> str:
    """
    Reorder item from supplier. Automatically adds item to inventory if not found.

    This tool checks inventory, adds new items if needed, updates stock levels,
    and records the supplier transaction.

    Args:
        item_name: Name of the item to reorder
        reorder_quantity: Quantity to reorder (must be > 0)
        request_date: Date of the request in ISO format (YYYY-MM-DD)
        unit_price: Unit price per unit (must be >= 0)
        category: Category for new items. Options: "paper", "specialty", "large_format", "product"
        min_stock_level: Minimum stock level for new items. Use 0 for auto-calculation (20% of reorder quantity)

    Returns:
        JSON string with reorder confirmation, inventory updates, and stock status

    Examples:
        reorder_item_tool("Cardstock", 1000, "2025-01-15", 0.15)
        reorder_item_tool("Custom Paper", 500, "2025-01-15", 0.25, category="specialty", min_stock_level=100)
    """
    # Convert 0 to None for auto-calculation
    min_level = min_stock_level if min_stock_level > 0 else None

    return reorder_item(
        item_name=item_name,
        reorder_quantity=reorder_quantity,
        request_date=request_date,
        unit_price=unit_price,
        category=category,
        min_stock_level=min_level,
    )


@tool
def calculate_reorder_quantity(
    current_stock: int, order_quantity: int, min_stock_level: int = 0
) -> int:
    """Calculate reorder quantity.

    Args:
        current_stock (int): quantity in stock
        min_stock_level (int): minimum stock level
        order_quantity (int): quantity ordered

    Returns:
        reorder_quantity (int): quantity to reorder
    """

    reorder_quantity = min_stock_level - (current_stock - order_quantity)

    if reorder_quantity < 0:
        reorder_quantity = 0

    return reorder_quantity


@tool
def get_inventory_snapshot(as_of_date: str) -> Dict[str, Any]:
    """
    Get a complete snapshot of all inventory items.

    Args:
        as_of_date: Date for the inventory snapshot (YYYY-MM-DD format)

    Returns:
        Dictionary with inventory snapshot data
    """
    inventory = get_all_inventory(as_of_date)

    # avoid this error :

    # avoid this error : TypeError: string indices must be integers, not 'str'
    # inventory = json.loads(inventory)

    print(f"Inventory as of {as_of_date}: {inventory}")

    # calculate total quantity of keys in dictionary
    total_quantity = sum(inventory[item] for item in inventory)

    # Identify any products that are below min stock level
    # low_stock_items = [        item for item in inventory if item["current_stock"] < item["min_stock_level"]    ]

    return {
        "inventory": inventory,
        "as_of_date": as_of_date,
        "total_quantity_inventory": total_quantity,
        # "total_items": len(inventory),
        # "low_stock_items": low_stock_items,
    }


@tool
def get_inventory_details_snapshot(request_date: str) -> str:
    """
    Get a complete snapshot of all inventory items as of a specific date.

    Provides comprehensive inventory health analysis including:
    - All items with current stock levels
    - Items below minimum stock (low stock alert)
    - Out of stock items
    - Overall inventory health score
    - Total inventory value

    Args:
        request_date: Date to check inventory (ISO format YYYY-MM-DD)

    Returns:
        JSON string with complete inventory snapshot and health metrics

    Examples:
        get_inventory_snapshot("2025-04-01")
    """
    try:
        # Get all inventory using the provided get_all_inventory function
        inventory_dict = get_all_inventory(request_date)

        if not inventory_dict:
            return json.dumps(
                {
                    "success": False,
                    "total_items": 0,
                    "inventory_items": [],
                    "low_stock_items": [],
                    "out_of_stock_items": [],
                    "inventory_health_score": 0.0,
                    "message": "No inventory data available for the specified date",
                },
                indent=2,
            )

        # Get inventory reference data (categories, min levels, prices)
        query = "SELECT * FROM inventory"
        inventory_df = pd.read_sql(query, db_engine)

        if inventory_df.empty:
            return json.dumps(
                {
                    "success": False,
                    "error": "Inventory reference table is empty",
                    "total_items": 0,
                },
                indent=2,
            )

        # Build comprehensive inventory snapshot
        inventory_items = []
        low_stock_items = []
        out_of_stock_items = []
        healthy_items = 0
        total_value = 0.0

        for item_name, current_stock in inventory_dict.items():
            # Find matching item in inventory reference
            item_info = inventory_df[
                inventory_df["item_name"].str.lower() == item_name.lower()
            ]

            if item_info.empty:
                continue

            item_data = item_info.iloc[0]
            min_stock_level = int(item_data["min_stock_level"])
            unit_price = float(item_data["unit_price"])
            category = item_data["category"]

            # Calculate stock metrics
            coverage_ratio = (
                current_stock / min_stock_level if min_stock_level > 0 else 0
            )
            stock_value = current_stock * unit_price
            total_value += stock_value

            # Determine stock status
            if current_stock == 0:
                stock_status = "out_of_stock"
                out_of_stock_items.append(
                    {
                        "item_name": item_name,
                        "min_stock_level": min_stock_level,
                        "unit_price": unit_price,
                        "category": category,
                    }
                )
            elif current_stock < min_stock_level:
                stock_status = "low_stock"
                shortage = min_stock_level - current_stock
                recommended_reorder = max(shortage, min_stock_level * 2)
                low_stock_items.append(
                    {
                        "item_name": item_name,
                        "current_stock": current_stock,
                        "min_stock_level": min_stock_level,
                        "shortage": shortage,
                        "recommended_reorder": recommended_reorder,
                        "estimated_cost": round(recommended_reorder * unit_price, 2),
                    }
                )
            else:
                stock_status = "healthy"
                healthy_items += 1

            # Add to inventory items list
            inventory_items.append(
                {
                    "item_name": item_name,
                    "current_stock": current_stock,
                    "min_stock_level": min_stock_level,
                    "unit_price": unit_price,
                    "category": category,
                    "stock_status": stock_status,
                    "coverage_ratio": round(coverage_ratio, 2),
                    "stock_value": round(stock_value, 2),
                }
            )

        # Calculate inventory health score
        total_items = len(inventory_items)
        health_score = (healthy_items / total_items * 100) if total_items > 0 else 0

        # Build summary message
        message = (
            f"Inventory health: {health_score:.1f}% - "
            f"{healthy_items} items healthy, "
            f"{len(low_stock_items)} low stock, "
            f"{len(out_of_stock_items)} out of stock"
        )

        # Sort items by stock status priority
        status_order = {"out_of_stock": 0, "low_stock": 1, "healthy": 2}
        inventory_items.sort(key=lambda x: status_order[x["stock_status"]])

        result = {
            "success": True,
            "request_date": request_date,
            "total_items": total_items,
            "inventory_items": inventory_items,
            "low_stock_items": low_stock_items,
            "out_of_stock_items": out_of_stock_items,
            "inventory_health_score": round(health_score, 2),
            "healthy_items_count": healthy_items,
            "low_stock_count": len(low_stock_items),
            "out_of_stock_count": len(out_of_stock_items),
            "total_inventory_value": round(total_value, 2),
            "message": message,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Error getting inventory snapshot: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def find_similar_past_quotes(search_terms: str, limit: int = 5) -> Dict[str, Any]:
    """
    Search historical quotes to help with pricing decisions.

    Args:
        search_terms: Keywords to search for in past quotes
        limit: Maximum number of results to return

    Returns:
        Dictionary containing historical quotes that match the search
    """
    # Parse search terms and create a list of strings
    search_terms = search_terms.split(",")
    search_terms = [term.strip() for term in search_terms]

    # print(f"Searching historical quotes for terms: {search_terms}")
    # print(f"{isinstance(search_terms, list)=}")
    quotes = search_quote_history(search_terms, 2)
    # print(quotes)
    return {"historical_quotes": quotes, "count": len(quotes)}


@tool
def check_delivery_timeline(
    item_name: str, reorder_quantity: int, request_date: str
) -> dict[str, Any]:
    """Check delivery timeline from supplier.

    Args:
        item_name (str): Name of the item.
        reorder_quantity (int): Quantity that needs to be reordered.
        request_date (str): Date of the request in ISO format.

    Returns:
        str: dictionary with delivery timeline information.
    """
    try:
        delivery_date = get_supplier_delivery_date(request_date, reorder_quantity)
        request_dt = datetime.fromisoformat(request_date.split("T")[0])
        delivery_dt = datetime.fromisoformat(delivery_date)
        days = (delivery_dt - request_dt).days

        return {
            "item_name": item_name,
            "request_date": request_date,
            "estimated_delivery_date": delivery_date,
            "estimated_days": days,
            "message": f"Delivery estimated in {days} business day(s) - by {delivery_date}",
        }

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@tool
def calculate_order_total(quotation_items: str) -> Dict[str, Any]:
    """Calculate total for an order using selling prices (includes 20% markup).

    This function calculates the total amount for an order by summing up the costs
    of all items using their SELLING PRICES (which include the 20% markup applied
    by calculate_single_item_quote).

    Args:
        quotation_items: JSON string containing a list of item dictionaries.
            Format: '[{"item_name": "Paper", "order_quantity": 100, "selling_price": 0.24}, ...]'

            Each item must have:
            - item_name (str): Name of the item
            - order_quantity (int): Quantity ordered
            - selling_price (float): REQUIRED - Price with 20% markup
            - unit_price (float): Optional - Cost price (for reference only)
            - category (str): Optional - Item category

    Returns:
        Dict[str, Any]: Dictionary containing:
            - success (bool): True if calculation succeeded
            - items_count (int): Number of items processed
            - items (list): List of items with details
            - total_quantity (int): Total units across all items
            - total_amount (float): Sum of all item totals (using selling_price)
            - message (str): Summary message
            - error (str): Error message if success=False

    Examples:
        >>> # Single item
        >>> items_str = '[{"item_name": "Paper", "order_quantity": 100, "selling_price": 0.24}]'
        >>> result = calculate_order_total(items_str)
        >>> result["total_amount"]
        24.0

        >>> # Multiple items
        >>> items_str = '''[
        ...     {"item_name": "Glossy paper", "order_quantity": 200, "selling_price": 0.24},
        ...     {"item_name": "Cardstock", "order_quantity": 100, "selling_price": 0.18}
        ... ]'''
        >>> result = calculate_order_total(items_str)
        >>> result["total_amount"]  # (200 × 0.24) + (100 × 0.18) = 66.0
        66.0

    Important:
        - Input MUST be a valid JSON string representing a list
        - ALWAYS uses selling_price for calculations (NOT unit_price)
        - selling_price must include the 20% markup
        - Returns error if JSON is invalid or list is empty
    """
    try:
        # Validate input type
        if not isinstance(quotation_items, str):
            return {
                "success": False,
                "error": f"Input must be a JSON string, got {type(quotation_items).__name__}",
                "items_count": 0,
                "items": [],
                "total_amount": 0.0,
            }

        # Parse JSON string to list
        try:
            items = json.loads(quotation_items)
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Invalid JSON string: {str(e)}",
                "items_count": 0,
                "items": [],
                "total_amount": 0.0,
            }

        # Validate parsed result is a list
        if not isinstance(items, list):
            return {
                "success": False,
                "error": f"JSON must represent a list, got {type(items).__name__}",
                "items_count": 0,
                "items": [],
                "total_amount": 0.0,
            }

        # Check for empty list
        if not items:
            return {
                "success": False,
                "error": "No items provided in the list",
                "items_count": 0,
                "items": [],
                "total_amount": 0.0,
            }

        total_amount = Decimal("0")
        total_quantity = 0
        order_items = []

        for idx, item_data in enumerate(items):
            # Validate each item is a dictionary
            if not isinstance(item_data, dict):
                return {
                    "success": False,
                    "error": f"Item at index {idx} is not a dict: {type(item_data).__name__}",
                    "items_count": 0,
                    "items": [],
                    "total_amount": 0.0,
                }

            # Extract required fields
            item_name = item_data.get("item_name", "")
            quantity = item_data.get("order_quantity", 0)

            # Use selling_price for calculations (includes 20% markup)
            # Fallback to unit_price only if selling_price is not provided
            selling_price = item_data.get("selling_price") or item_data.get(
                "unit_price", 0
            )

            if not selling_price:
                return {
                    "success": False,
                    "error": f"Item '{item_name}' at index {idx} missing both selling_price and unit_price",
                    "items_count": 0,
                    "items": [],
                    "total_amount": 0.0,
                }

            selling_price_decimal = Decimal(str(selling_price))
            item_total = selling_price_decimal * Decimal(str(quantity))

            order_items.append(
                {
                    "item_name": item_name,
                    "category": item_data.get("category", ""),
                    "quantity": quantity,
                    "unit_price": float(item_data.get("unit_price", 0)),
                    "selling_price": float(selling_price_decimal),
                    "total": float(item_total),
                }
            )

            total_amount += item_total
            total_quantity += quantity

        return {
            "success": True,
            "items_count": len(order_items),
            "items": order_items,
            "total_quantity": total_quantity,
            "total_amount": float(total_amount),
            "message": f"Calculated total for {len(order_items)} item(s): Total Quantity: {total_quantity}, Total Amount: ${float(total_amount):,.2f}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "items_count": 0,
            "items": [],
            "total_amount": 0.0,
        }


def calculate_multi_item_quote(quotation_items: str) -> Dict[str, Any]:
    """Calculate quote for multi-item order.

    Args:
        quotation_items (str): JSON string with item details.

    Returns:
        str: dictionary with quote information.
    """
    data = json.loads(quotation_items)
    # data = order_details
    # data = ast.literal_eval(order_details)
    # items = order_details["order_details"]["items"]

    # Look for any key containing 'items'
    for key, value in data.items():
        if "items" in key.lower() and isinstance(value, list):
            items_key = key
            break

    items = data.get(items_key, [])
    # print(f"items: \n{items}")

    # items = data["query_items"]
    print(f"items: \n{items}")

    try:

        if not items:
            return json.dumps({"error": "No items provided"}, indent=2)

        total_amount = Decimal("0")
        markup = Decimal("1.2")  # 20% markup for quotes
        quote_items = []

        for item_data in items:
            item_name = item_data["item_name"]
            quantity = item_data["order_quantity"]
            unit_price = Decimal(str(item_data["unit_price"]))
            selling_price = Decimal(str(item_data["unit_price"])) * markup
            item_total = unit_price * Decimal(str(quantity)) * markup

            quote_items.append(
                {
                    "item_name": item_name,
                    "quantity": quantity,
                    "unit_price": float(unit_price),
                    "selling_price": float(selling_price),
                    "total": float(item_total),
                    "message": f"Item: {item_name}, Quantity: {quantity}, Unit Price: ${unit_price:.2f}, Selling Price: ${selling_price:.2f}, Total: ${item_total:.2f}",
                }
            )

            total_amount += item_total

        return json.dumps(
            {
                "success": True,
                "items_count": len(quote_items),
                "items": quote_items,
                # "total_quantity": sum(item["order_quantity"] for item in data["items"]),
                "total_amount": float(total_amount),
                # "message": f"Quote prepared: {len(quote_items)} item(s), Total: ${total_amount:.2f}",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def calculate_single_item_quote(
    item_name: str, order_quantity: int, unit_price: float
) -> str:
    """
    Calculate quote for a single item.

    Args:
        item_name (str): Name of the item.
        order_quantity (int): Quantity of the item to order.
        unit_price (float): Unit price of the item.

    Returns:
        JSON string with quote information including markup pricing

    Examples:
        calculate_single_item_quote('{"item_name": "Cardstock", "order_quantity": 500, "unit_price": 0.15}')
    """
    try:

        # ===== NEW VALIDATION BLOCK =====
        # Validate inputs
        if not item_name or item_name == "null":
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid item_name: cannot be null or empty",
                    "message": "Cannot calculate quote for invalid item",
                },
                indent=2,
            )

        if unit_price is None or unit_price <= 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid unit_price: {unit_price}",
                    "message": f"Cannot calculate quote for {item_name} without valid unit price",
                },
                indent=2,
            )

        if order_quantity <= 0:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Invalid order_quantity: {order_quantity}",
                    "message": "Order quantity must be greater than 0",
                },
                indent=2,
            )
        # ===== END VALIDATION BLOCK =====
        # Calculate quote with 20% markup
        markup = Decimal("1.2")  # 20% markup for quotes
        unit_price_decimal = Decimal(str(unit_price))
        quantity_decimal = Decimal(str(order_quantity))

        # Calculate prices
        cost_price = unit_price_decimal
        selling_price = unit_price_decimal * markup
        subtotal = cost_price * quantity_decimal
        total_amount = selling_price * quantity_decimal
        markup_amount = total_amount - subtotal

        result = {
            "success": True,
            "item_name": item_name,
            "order_quantity": order_quantity,
            "cost_price": float(cost_price),
            "selling_price": float(selling_price),
            "markup_percent": 20.0,
            "subtotal": float(subtotal),
            "markup_amount": float(markup_amount),
            "total_amount": float(total_amount),
            "message": (
                f"Quote for {item_name}: "
                f"{order_quantity} units @ ${selling_price:.2f} each = ${total_amount:.2f} "
            ),
        }

        print(f"Quote calculated: {result['message']}")
        return result

    except Exception as e:
        logger.exception(f"Error calculating single item quote: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def fulfill_multi_item_order(quote_details: str) -> Dict[str, Any]:
    """Fulfill multi-item order.

    Args:
        quote_details (str): dictionary with quote details.

        Example:
            fulfill_multi_item_order('{"quotation_items": [{"item_name": "Cardstock", "quantity": 500, "unit_price": 0.15}]}')

    Returns:
        str: dictionary with order fulfillment information.
    """
    try:
        data = json.loads(quote_details)

        items = data.get("quotation_items", [])
        # items = data["items"]

        if not items:
            return json.dumps({"error": "No items provided"}, indent=2)

        # Validate all items first
        insufficient_items = []
        for item_data in items:
            item_name = item_data.get("item_name")
            quantity = item_data.get("quantity", 0)

            query = (
                f"SELECT * FROM inventory WHERE LOWER(item_name) = LOWER('{item_name}')"
            )
            result = pd.read_sql(query, db_engine)

            if result.empty:
                return json.dumps(
                    {"success": False, "error": f"Item not found: {item_name}"},
                    indent=2,
                )

            item = result.iloc[0].to_dict()
            if item["current_stock"] < quantity:
                insufficient_items.append(
                    {
                        "item_name": item_name,
                        "requested": quantity,
                        "available": item["current_stock"],
                    }
                )

        if insufficient_items:
            return json.dumps(
                {
                    "success": False,
                    "error": "Insufficient stock for one or more items",
                    "insufficient_items": insufficient_items,
                },
                indent=2,
            )

        # Process order
        order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
        total_amount = Decimal("0")
        processed_items = []
        reorders_needed = []

        for item_data in items:
            item_name = item_data["item_name"]
            quantity = item_data["order_quantity"]
            unit_price = Decimal(str(item_data["selling_price"]))
            item_total = unit_price * Decimal(str(quantity))

            query = (
                f"SELECT * FROM inventory WHERE LOWER(item_name) = LOWER('{item_name}')"
            )
            result = pd.read_sql(query, db_engine)
            item = result.iloc[0].to_dict()

            new_stock = int(item["current_stock"]) - quantity

            update_query = f"UPDATE inventory SET current_stock = {new_stock} WHERE LOWER(item_name) = LOWER('{item_name}')"
            with db_engine.connect() as conn:
                conn.execute(text(update_query))
                conn.commit()

            insert_query = f"""INSERT INTO transactions (item_name, transaction_type, units, price, transaction_date) 
                            VALUES ('{item_name}', 'sales', {quantity}, {float(item_total)}, '{state['request_date']}')"""
            with db_engine.connect() as conn:
                conn.execute(text(insert_query))
                conn.commit()

            processed_items.append(
                {
                    "item_name": item_name,
                    "quantity": quantity,
                    "unit_price": float(unit_price),
                    "total": float(item_total),
                    "new_stock": new_stock,
                }
            )

            if new_stock < item["min_stock_level"]:
                reorders_needed.append(
                    {
                        "item_name": item_name,
                        "current_stock": new_stock,
                        "min_level": item["min_stock_level"],
                    }
                )
            else:
                reorders_needed.append(None)

            total_amount += item_total

        return json.dumps(
            {
                "success": True,
                "order_id": order_id,
                "customer_job": state["customer_job"],  # customer_job,
                "event_type": state["event_type"],  # event_type,
                "items_count": len(processed_items),
                "items": processed_items,
                "total_amount": float(total_amount),
                "tracking_number": f"TRK{uuid.uuid4().hex[:12].upper()}",
                "reorders_needed": reorders_needed,
                "message": f"Order {order_id} confirmed: {len(processed_items)} item(s), Total: ${total_amount:.2f}",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def generate_order_id() -> str:
    """Generate a unique order ID.

    Returns:
        str: Unique order ID.
    """
    return f"ORD_{uuid.uuid4().hex[:8].upper()}"


@tool
def generate_tracking_number() -> str:
    """Generate a unique tracking number.

    Returns:
        str: Unique tracking number.
    """

    # Mock implementation of issuing tracking number
    return f"TRK{uuid.uuid4().hex[:12].upper()}"


@tool
def convert_ream_to_sheets(reams: int, sheets_per_ream: int = 500) -> int:
    """Convert reams to sheets.

    Args:
        reams (int): Number of reams.
        sheets_per_ream (int): Number of sheets per ream. Default is 500.

    Returns:
        int: Total number of sheets.
    """
    return reams * sheets_per_ream


@tool
def fulfill_single_item_order(
    item_name: str,
    category: str,
    order_quantity: int,
    unit_price: float,
    selling_price: float,
    request_date: str,
    reorder_quantity: int = 0,
) -> str:
    """
    Fulfill a single item order by validating stock, updating inventory, and recording transaction.

    Args:
        item_name: Name of the item to order.
        category: Category of the item. (required)
        order_quantity: Quantity of the item to order.
        unit_price: Cost price of the item used for reordering transaction.
        selling_price: Selling price of the item used for sales transaction.
        request_date: Date of the order request (ISO format). Defaults to current date.
        reorder_quantity: Quantity of the item to reorder. Defaults to 0.

    Returns:
        JSON string with order fulfillment confirmation or error details

    Examples:
        fulfill_single_item_order(
            item_name="Cardstock",
            order_quantity=500,
            unit_price=0.15,
            selling_price=0.18,
            request_date="2024-10-01"
        )
    """
    try:
        # Check if reorder is needed
        if reorder_quantity > 0:
            reorder_result = reorder_item_tool(
                item_name=item_name,
                category=category,
                reorder_quantity=reorder_quantity,
                request_date=request_date,
                unit_price=unit_price,
            )
            print(f"Reorder placed: {reorder_result}")

        # Check if item exists in inventory
        query = f"SELECT * FROM inventory WHERE LOWER(item_name) = LOWER('{item_name}')"
        result = pd.read_sql(query, db_engine)

        if result.empty:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Item not found in inventory: {item_name}",
                    "message": "Cannot fulfill order for non-existent item",
                    "found": False,
                },
                indent=2,
            )

        item = result.iloc[0].to_dict()

        # Validate sufficient stock
        if item["current_stock"] < order_quantity:
            return json.dumps(
                {
                    "success": False,
                    "error": "Insufficient stock",
                    "item_name": item_name,
                    "requested": order_quantity,
                    "available": item["current_stock"],
                    "message": f"Cannot fulfill order: requested {order_quantity} units, but only {item['current_stock']} available",
                },
                indent=2,
            )

        # Calculate order totals
        unit_price = Decimal(str(selling_price))
        quantity_decimal = Decimal(str(order_quantity))
        item_total = unit_price * quantity_decimal

        # Update inventory
        new_stock = int(item["current_stock"]) - order_quantity
        update_query = f"UPDATE inventory SET current_stock = {new_stock} WHERE LOWER(item_name) = LOWER('{item_name}')"

        with db_engine.connect() as conn:
            conn.execute(text(update_query))
            conn.commit()

        # print(f"Inventory updated: {item_name} stock {item['current_stock']} -> {new_stock}")

        # Record transaction
        insert_query = f"""
            INSERT INTO transactions (item_name, transaction_type, units, price, transaction_date) 
            VALUES ('{item_name}', 'sales', {order_quantity}, {float(item_total)}, '{request_date}')
        """

        with db_engine.connect() as conn:
            conn.execute(text(insert_query))
            conn.commit()

        # print(f"Transaction recorded: {order_quantity} units of {item_name} sold for ${item_total:.2f}")

        """
        
        # Generate order ID and tracking number
        order_id = f"ORD_{uuid.uuid4().hex[:8].upper()}"
        tracking_number = f"TRK{uuid.uuid4().hex[:12].upper()}"
        
        
        reorder_needed = None
        if new_stock < item["min_stock_level"]:
            shortage = item["min_stock_level"] - new_stock
            reorder_needed = {
                "item_name": item_name,
                "current_stock": new_stock,
                "min_level": item["min_stock_level"],
                "shortage": shortage,
                "recommended_order": max(shortage, item["min_stock_level"] * 2),
                "alert": f"⚠️ Stock below minimum! Current: {new_stock}, Minimum: {item['min_stock_level']}",
            }
            logger.warning(f"Reorder needed for {item_name}: {reorder_needed['alert']}")
        """
        # Build result
        result = {
            "success": True,
            "order_date": request_date,
            "item": {
                "item_name": item_name,
                "quantity": order_quantity,
                "unit_price": float(unit_price),
                "total": float(item_total),
                "previous_stock": item["current_stock"],
                "new_stock": new_stock,
            },
            "total_amount": float(item_total),
            "quantity_reordered": reorder_quantity,
            "fulfillment_status": "Completed",
            "message": f"✓ Order confirmed: {order_quantity} units of {item_name}, Total: ${item_total:.2f}",
        }

        # print(f"Order fulfilled successfully: {result['message']}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception(f"Error fulfilling single item order: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@tool
def compare_delivery_dates(
    expected_delivery_date: str, estimated_delivery_date: str
) -> Dict[str, Any]:
    """Compare requested and estimated delivery dates.

    Args:
        expected_delivery_date (str): Expected delivery date in ISO format (YYYY-MM-DD).
        estimated_delivery_date (str): Estimated delivery date in ISO format (YYYY-MM-DD).

    Returns:
        Dict[str, Any]: Dictionary with comparison results.

        Examples:
            compare_delivery_dates("Cardstock", "2024-10-15", "2024-10-12")
            compare_delivery_dates("A3 Paper", "2024-10-15", "2024-10-18")

        Example Results:
            {
            "delivery_date_comparison": {
                'expected_delivery_date': '2024-10-15',
                'estimated_delivery_date': '2024-10-12',
                'status': 'On Time',
                'message': '✓ Estimated delivery date 2024-10-12 meets the requested date 2024-10-15.'
                }
            }
            {
            "delivery_date_comparison": {
                'expected_delivery_date': '2024-10-15',
                'estimated_delivery_date': '2024-10-18',
                'status': 'Delayed',
                'message': '⚠️ Estimated delivery date 2024-10-18 is 3 day(s) later than the requested date 2024-10-15.'
                }
            }
    """
    try:
        exp_date = datetime.fromisoformat(expected_delivery_date)
        est_date = datetime.fromisoformat(estimated_delivery_date)

        if est_date <= exp_date:
            status = "On Time"
            message = f"✓ Estimated delivery date {estimated_delivery_date} meets the requested date {expected_delivery_date}."
        else:
            status = "Delayed"
            days_late = (est_date - exp_date).days
            message = f"⚠️ Estimated delivery date {estimated_delivery_date} is {days_late} day(s) later than the requested date {expected_delivery_date}."

        return {
            "delivery_date_comparison": {
                "expected_delivery_date": expected_delivery_date,
                "estimated_delivery_date": estimated_delivery_date,
                "status": status,
                "message": message,
            }
        }

    except Exception as e:
        return {"error": str(e)}


class QuotationItem(TypedDict):
    """Type definition for quotation line items."""

    item_name: str
    order_quantity: int
    price: float
    total: float


class OrderConfirmation(TypedDict, total=False):
    """Type definition for order confirmation dictionary."""

    request_date: str
    original_request: str
    customer_job: str
    event_type: str
    need_size: OrderSize | str
    order_id: str
    quotation: str
    quotation_items: list[QuotationItem]
    order_delivery_date: str
    total_amount: float
    tracking_number: str


def format_order_confirmation_email(order_data: dict[str, Any]) -> str:
    """
    Format an order confirmation dictionary into a professional email.

    This function takes a comprehensive order dictionary and converts it into
    a well-formatted, human-readable email suitable for sending to customers.
    All data is preserved exactly as provided without modification.

    Args:
        order_data: Dictionary containing order confirmation details with keys:
            - request_date: ISO date string (YYYY-MM-DD)
            - original_request: Customer's original request text
            - customer_job: Customer's job title/role
            - event_type: Type of event (conference, meeting, etc.)
            - need_size: Order size (small/medium/large)
            - order_id: Unique order identifier
            - quotation: Full quotation text
            - quotation_items: List of line items with details
            - unavailable_items: List of items that cannot be ordered (optional)
            - order_delivery_date: Expected delivery date
            - total_amount: Final total after discounts
            - tracking_number: Shipment tracking number (if available)

    Returns:
        Formatted email string ready to send to customer

    Examples:
        >>> order = {
        ...     "request_date": "2025-04-04",
        ...     "order_id": "ORD_ABC123",
        ...     "customer_job": "event manager",
        ...     "total_amount": 1500.00,
        ...     # ... other fields
        ... }
        >>> email = format_order_confirmation_email(order)
        >>> print(email)
    """
    # Extract and format header information
    order_id = order_data.get("order_id", "N/A")
    request_date = _format_date(order_data.get("request_date", ""))
    customer_job = order_data.get("customer_job", "Valued Customer")
    event_type = order_data.get("event_type", "order")
    unavailable_items = order_data.get("unavailable_items", [])

    # Build email header
    email_lines = [
        "=" * 80,
        "MUNDER DIFFLIN PAPER SUPPLY COMPANY",
        "Order Confirmation",
        "=" * 80,
        "",
        f"Order ID: {order_id}",
        f"Order Date: {request_date}",
        f"Customer: {customer_job.title()}",
        f"Event Type: {event_type.title()}",
        "",
        "=" * 80,
    ]

    # Add original request section
    original_request = order_data.get("original_request", "")
    if original_request:
        email_lines.extend(
            [
                "ORIGINAL REQUEST",
                "=" * 80,
                "",
                _wrap_text(original_request, width=76),
                "",
                "=" * 80,
            ]
        )

    # Add quotation section
    quotation = order_data.get("quotation", "")
    if quotation:
        email_lines.extend(
            [
                "QUOTATION SUMMARY",
                "=" * 80,
                "",
                _wrap_text(quotation, width=76),
                "",
                "=" * 80,
            ]
        )

    # Add detailed line items table
    quotation_items = order_data.get("quotation_items", [])
    if quotation_items:
        email_lines.extend(
            [
                "ORDER DETAILS",
                "=" * 80,
                "",
                _format_line_items_table(quotation_items),
                "",
                "=" * 80,
            ]
        )

    # Add financial summary
    total_amount = order_data.get("total_amount", 0.0)
    need_size = order_data.get("need_size", "")

    # Extract order size if it's an enum
    if isinstance(need_size, Enum):
        order_size = need_size.value
    else:
        order_size = str(need_size) if need_size else "Standard"

    email_lines.extend(
        [
            "FINANCIAL SUMMARY",
            "=" * 80,
            "",
            f"Order Size Classification: {order_size.title()}",
            f"Final Total (After Discounts): ${total_amount:,.2f}",
            "",
        ]
    )

    # Calculate subtotal from line items if available
    if quotation_items:
        subtotal = sum(item.get("total", 0.0) for item in quotation_items)
        discount_amount = subtotal - total_amount

        if discount_amount > 0:
            discount_percent = (discount_amount / subtotal * 100) if subtotal > 0 else 0
            email_lines.extend(
                [
                    f"Subtotal: ${subtotal:,.2f}",
                    f"Discount Applied: -${discount_amount:,.2f} ({discount_percent:.1f}%)",
                    f"Total Amount: ${total_amount:,.2f}",
                    "",
                ]
            )

    email_lines.append("=" * 80)

    # Add unavailable items section if any
    if unavailable_items:
        email_lines.extend(
            [
                "ITEMS NOT AVAILABLE",
                "=" * 80,
                "",
                "The following items could not be included in your order:",
                "",
            ]
        )
        
        for item in unavailable_items:
            query = item.get("query", "Unknown Item")
            order_quantity = item.get("order_quantity", 0)
            reason = item.get("reason", "Not available in catalog")
            
            email_lines.append(f"• {order_quantity:,} {query} - {reason}")
        
        email_lines.extend(
            [
                "",
                "We apologize for the inconvenience. Please contact us if you would",
                "like assistance finding alternative products or substitutes.",
                "",
                "=" * 80,
            ]
        )

    # Add delivery information
    delivery_date = _format_date(order_data.get("order_delivery_date", ""))
    tracking_number = order_data.get("tracking_number", "")

    email_lines.extend(
        [
            "DELIVERY INFORMATION",
            "=" * 80,
            "",
        ]
    )

    if delivery_date:
        email_lines.append(f"Expected Delivery Date: {delivery_date}")

    if tracking_number:
        email_lines.append(f"Tracking Number: {tracking_number}")
    else:
        email_lines.append("Tracking Number: Will be provided upon shipment")

    email_lines.extend(
        [
            "",
            "Your order will be carefully packaged and shipped to ensure",
            "safe delivery. You will receive tracking updates via email.",
            "",
            "=" * 80,
        ]
    )

    # Add footer
    email_lines.extend(
        [
            "NEXT STEPS",
            "=" * 80,
            "",
            "✓ Your order has been confirmed and is being processed",
            "✓ You will receive shipment notification with tracking details",
            "✓ Contact us at orders@munderdifflin.com for any questions",
            "",
            "=" * 80,
            "",
            "Thank you for choosing Munder Difflin Paper Supply Company!",
            "",
            "We appreciate your business and look forward to serving you.",
            "",
            "Best regards,",
            "The Munder Difflin Team",
            "",
            "=" * 80,
            "",
            "Munder Difflin Paper Supply Company",
            "Phone: 1-800-MUNDER-1 | Email: orders@munderdifflin.com",
            "Web: www.munderdifflin.com",
            "",
        ]
    )

    return "\n".join(email_lines)


def _format_date(date_str: str) -> str:
    """
    Format ISO date string to human-readable format.

    Args:
        date_str: ISO format date string (YYYY-MM-DD)

    Returns:
        Formatted date string (e.g., "April 4, 2025")
        Falls back to original string if parsing fails
    """
    if not date_str:
        return "N/A"

    try:
        # Handle both date-only and datetime formats
        date_part = date_str.split("T")[0] if "T" in date_str else date_str
        dt = datetime.fromisoformat(date_part)
        return dt.strftime("%B %d, %Y")
    except (ValueError, AttributeError):
        return date_str


def _wrap_text(text: str, width: int = 76) -> str:
    """
    Wrap text to specified width while preserving paragraphs.

    Args:
        text: Text to wrap
        width: Maximum line width (default: 76 characters)

    Returns:
        Wrapped text with proper line breaks
    """
    import textwrap

    # Split into paragraphs
    paragraphs = text.split("\n")

    # Wrap each paragraph individually
    wrapped_paragraphs = []
    for para in paragraphs:
        if para.strip():
            wrapped = textwrap.fill(
                para.strip(),
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_paragraphs.append(wrapped)
        else:
            wrapped_paragraphs.append("")

    return "\n".join(wrapped_paragraphs)


def _format_line_items_table(items: list[QuotationItem]) -> str:
    """
    Format line items as a professional table.

    Args:
        items: List of quotation items with name, quantity, price, and total

    Returns:
        Formatted table string
    """
    if not items:
        return "No items in order"

    lines = []

    # Table header
    lines.append(f"{'Item Name':<30} {'Quantity':>12} {'Unit Price':>12} {'Total':>12}")
    lines.append("-" * 80)

    # Table rows
    for item in items:
        item_name = item.get("item_name", "Unknown Item")
        quantity = item.get("order_quantity", 0)
        price = item.get("price", 0.0)
        total = item.get("total", 0.0)

        # Format numbers
        quantity_str = f"{quantity:,}"
        price_str = f"${price:.2f}"
        total_str = f"${total:,.2f}"

        lines.append(
            f"{item_name:<30} {quantity_str:>12} {price_str:>12} {total_str:>12}"
        )

    # Table footer with total
    lines.append("-" * 80)
    subtotal = sum(item.get("total", 0.0) for item in items)
    lines.append(f"{'SUBTOTAL':<30} {'':<12} {'':<12} ${subtotal:>11,.2f}")

    return "\n".join(lines)


# Agents
class OrderProcessingAgent(ToolCallingAgent):
    """Agent responsible for processing customer order requests."""

    def __init__(self, model: Any):
        """
        Initialize the Order Processing Agent.

        Args:
            model: LLM model instance (HfApiModel, LiteLLMModel, etc.)
        """
        super().__init__(
            tools=[semantic_search_catalog, convert_ream_to_sheets],
            model=model,
            name="order_processing_agent",
            description="""
You are an Order Processing Agent for Munder Difflin paper supply company.

Your responsibilities:
- Understand customer order requests
- Identify items needed
- Identify quantities needed
- Identify delivery dates if required
- Identify paper sizes
- Identify any special instructions or notes for items if required
- VALIDATE that semantic search results are semantically correct

Use the semantic_search_catalog tool to find items in the catalog. When using the tool, provide a search query. For example if the customer requires "300 sheets of photo paper", use semantic_search_catalog with "photo paper" as the search term. 

Use the convert_ream_to_sheets tool to convert reams to sheets when the customer specifies quantity in reams.

Examples:  
Request: "I need 5 reams of A4 matte paper for an event"
Response: convert_ream_to_sheets with 5 as the number of reams. semantic_search_catalog with "A4 matte paper" as the search query.

Request: "I need 300 sheets of large photo paper for my wedding"
Response: semantic_search_catalog with "photo paper" as the search query 

Request: "I need 200 sheets of A4 glossy paper for a conference"
Response: semantic_search_catalog with "glossy paper" as the search query

---

CRITICAL VALIDATION RULES:

1. If semantic_search_catalog returns a match, VERIFY the match makes semantic sense
2. Check if the matched item's category is appropriate for the query:
   - Example: "tickets" should NOT match "paper" or "specialty paper" items
   - Example: "balloons" should NOT match any paper products
   - Example: "posters" (finished product) is DIFFERENT from "Poster paper" (raw material)
3. Only set found=True if:
   - Confidence is "high" or "medium" (the tool already filters this)
   - The matched item logically corresponds to what was requested
   - The category makes sense (e.g., paper products for paper requests)
4. If the semantic_search_catalog tool returns found=False, respect that decision
5. If you receive a match but it doesn't make semantic sense, override and set found=False

EXAMPLES OF INVALID MATCHES TO REJECT:
- Query: "tickets" → Matched: "Sticky notes" ❌ INVALID (different products) → Set found=False
- Query: "balloons" → Matched: "Colored paper" ❌ INVALID (not related) → Set found=False
- Query: "posters" → Matched: "Poster paper" ❌ INVALID (raw vs finished) → Set found=False
- Query: "decorations" → Matched: any paper product ❌ INVALID → Set found=False

EXAMPLES OF VALID MATCHES TO ACCEPT:
- Query: "flyers" → Matched: "Flyers" ✅ VALID (exact match) → Set found=True
- Query: "A4 paper" → Matched: "A4 paper" ✅ VALID (exact match) → Set found=True
- Query: "glossy paper" → Matched: "Glossy paper" ✅ VALID (exact match) → Set found=True
- Query: "cardstock" → Matched: "Cardstock" ✅ VALID (exact match) → Set found=True

---

Always use boolean True or boolean False for the 'found' field based on:
1. Semantic search confidence level (tool already validates this)
2. Logical semantic correctness of the match
3. Category appropriateness
4. Common sense - does the match actually correspond to what was requested?

Available tools:
1. semantic_search_catalog: Search catalog by item name keyword (returns validated, high-confidence matches)
2. convert_ream_to_sheets: Convert reams to sheets (1 ream = 500 sheets)

Only select one 'item_name' for each 'query' in the order request that best matches the customer's needs. Never have 2 query_items with the same 'query' in the order details.

Important Notes about quantity:
- 1 ream = 500 sheets
- If the quantity requested is 10 reams then the order_quantity is 5000 sheets.
- If the order_quantity requested is 5 reams then the order_quantity is 2500 sheets.

You must calculate the order_quantity in sheets based on the customer's request. Use the convert_ream_to_sheets tool if needed.
""",
        )

    def run(self, query: str, request_date: str = None) -> Dict[str, Any]:
        """
        Process a customer query and parse order details.

        Args:
            query (str): Customer's order request.

        Returns:
            Dict[str, Any]: Agent's response after processing the query.
        """

        return super().run(query)


class InventoryAgent(ToolCallingAgent):
    """
    Inventory management agent that checks stock levels and identifies items to reorder.

    Responsibilities:
    - check current stock levels
    - check delivery timeline for items needing reorder
    - place reorder for items below minimum stock levels or with insufficient stock to fulfill an order
    - provide detailed stock status and availability
    - calculate reorder quantities
    - compare requested vs estimated delivery dates
    """

    def __init__(self, model: Any):
        """
        Initialize the Inventory Agent.

        Args:
            model: LLM model instance (HfApiModel, LiteLLMModel, etc.)
        """
        super().__init__(
            tools=[
                check_current_stock,
                calculate_reorder_quantity,
                check_delivery_timeline,
                compare_delivery_dates,
                get_inventory_snapshot,
            ],
            model=model,
            name="inventory_agent",
            description="""
Inventory Management Agent for Munder Difflin Paper Supply

RESPONSIBILITIES:
- Check stock levels and calculate reorder quantities
- Place reorders for items needing replenishment
- Provide detailed stock status and availability
- Determine delivery timelines for reordered items
- Compare estimated delivery dates with expected delivery dates

WORKFLOW (Process items first, then compare dates):

FOR EACH VALID ITEM (where found = True/true):
1. check_current_stock(item_name, request_date, found) → get current stock levels
2. calculate_reorder_quantity(current_stock, order_quantity, min_stock_level) → determine reorder amount
3. check_delivery_timeline(item_name, reorder_quantity, request_date) → get estimated_delivery_date

AFTER PROCESSING ALL ITEMS:
4. Identify the LATEST (worst-case) estimated_delivery_date from all processed items
5. compare_delivery_dates(expected_delivery_date, latest_estimated_date) → perform comparison
   - expected_delivery_date: Extract from order_details['expected_delivery_date']
   - latest_estimated_date: The LATEST date from step 3 across ALL items
6. get_inventory_snapshot(request_date) → retrieve current inventory data snapshot 

ITEM VALIDATION RULE:
✅ ONLY process items where: found = True/true AND item_name != 'N/A'/'Not Found'/'None'
❌ SKIP items where: found = False/false OR item_name = 'N/A'/'Not Found'/'None'

TOOLS:
1. check_current_stock(item_name, request_date, found)
   - Check current inventory levels
   - Skip if item_name is "Not found"/"None" OR found is False
   
2. calculate_reorder_quantity(current_stock, order_quantity, min_stock_level)
   - Calculate optimal reorder amount
   - Only call for items that CAN be ordered (found = True)
   
3. check_delivery_timeline(item_name, reorder_quantity, request_date)
   - Estimate delivery date for reordered items
   - Returns: estimated_delivery_date (str in format 'YYYY-MM-DD')
   - Store this date for each item to find the latest
   
4. compare_delivery_dates(expected_delivery_date, estimated_delivery_date)
   - Compare expected vs estimated delivery dates
   - Call EXACTLY ONCE after processing ALL items
   - Use order_details['expected_delivery_date'] for first argument
   - Use LATEST estimated_delivery_date from all items for second argument
   
5. get_inventory_snapshot(request_date)
   - Retrieve current inventory data snapshot

CRITICAL INSTRUCTIONS:
- You MUST call compare_delivery_dates exactly once
- Call it AFTER all items have been processed
- Compare against the LATEST (worst-case) estimated delivery date
- The expected_delivery_date is in order_details passed to you

OUTPUT FORMAT:
{
    "inventory_items": [
        {
            "item_name": str,
            "order_quantity": int,
            "current_stock": int,
            "reorder_quantity": int,
            "min_stock_level": int,
            "estimated_delivery_date": str
        },
        ...
    ],
    "delivery_date_comparison": {
        "expected_delivery_date": str,
        "estimated_delivery_date": str,  # Latest from all items
        "status": str,  # From compare_delivery_dates result
        "message": str  # From compare_delivery_dates result
    },
    "inventory_snapshot": {
        ...  # Data from get_inventory_snapshot tool
    }
    
}

Do not use query search string, only use the item_name provided in the order details.
Apply validation rule to ALL tool calls.
""",
        )

    def run(self, query: str, order_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a customer query related to inventory.

        Args:
            query (str): Customer's inventory-related question or request.

        Returns:
            Dict[str, Any]: Agent's response after processing the query.
        """

        return super().run(query, additional_args={"order_details": order_details})


class QuotingAgent(ToolCallingAgent):
    """
    Quote generation agent that calculates quotes with bulk discounts.

    Responsibilities:
    - Calculate accurate quotes for single or multiple items
    - Apply volume-based bulk discounts
    - Provide detailed price breakdowns
    """

    def __init__(self, model: Any):
        """
        Initialize the Pricing Agent.

        Args:
            model: LLM model instance (HfApiModel, LiteLLMModel, etc.)
        """
        super().__init__(
            tools=[
                # calculate_multi_item_quote,
                find_similar_past_quotes,
                calculate_single_item_quote,
                calculate_order_total,
                calculate_bulk_discount,
            ],
            model=model,
            name="quoting_agent",
            description="""
## Role and Identity
You are a **Professional Quotation Specialist** responsible for generating accurate, customer-friendly price quotes for paper products and supplies.

---
⚠️ CRITICAL: BLANK QUOTE HANDLING ⚠️

BEFORE doing ANYTHING else, CHECK if ANY items can be ordered:

BLANK QUOTE CONDITIONS (return immediately without calling ANY tools):
1. ALL items have 'found' = False/false OR
2. ALL items have item_name = 'N/A' OR 'Not Found' OR 'None' OR
3. inventory_items list is empty OR
4. NO valid items exist in order_details['query_items']

IF ANY of above conditions are TRUE:
✋ STOP - Do NOT call ANY tools
✋ Return immediately with BLANK QUOTE:

{
    "quotation_text": "Thank you for your inquiry. Unfortunately, none of the items you requested are currently available in our catalog. We apologize that we cannot fulfill your order at this time. Please feel free to contact us if you'd like to explore alternative products.",
    "order_id": "[order_id from order_details]",
    "items_count": 0,
    "quotation_items": [],
    "subtotal": 0.0,
    "discount_percent": 0.0,
    "discount_amount": 0.0,
    "final_total": 0.0
}

IF at least ONE item has 'found' = True AND valid item_name:
✓ Proceed with normal quoting process using tools

---

## Core Responsibilities

### 1. Quote Generation
- Calculate precise quotes with 20% markup applied to all inventory prices
- Apply volume-based bulk discounts based on total quantity
- Apply order size modifiers for large orders
- Round totals to customer-friendly numbers when appropriate

### 2. Strategic Pricing Philosophy
- Balance profitability with customer satisfaction
- Encourage larger orders through discount structures
- Highlight savings opportunities transparently

### 3. Customer Communication
- Provide clear, detailed quote explanations
- Break down costs showing base price → markup → discounts → final total
- Only mention discounts when discount_amount > $0
- Maintain a warm, professional, helpful tone

---

⚠️ CRITICAL PRE-FLIGHT VALIDATION (DO THIS FIRST):

Before ANY calculations:
1. Count items where found=True in order_details['query_items']
2. Identify ALL unavailable items (found=False):
   - For EACH item where found=False, extract:
     * query (original search term)
     * order_quantity (requested quantity)
     * message (reason why not found)
   - Store in unavailable_items list with format:
     {"query": query, "order_quantity": qty, "reason": message}

3. If count = 0 (NO available items):
   - DO NOT call ANY calculation tools
   - IMMEDIATELY return final_answer with:
     * quotation_items: []
     * unavailable_items: [list of ALL requested items]
     * subtotal: 0.0
     * discount_percent: 0.0
     * discount_amount: 0.0
     * final_total: 0.0
     * quotation_text: MUST include:
       - Apology for unavailable items
       - List ALL items that cannot be ordered
       - Offer to help find alternatives
   - STOP HERE - Do not proceed to Phase 1

4. If count > 0 (SOME available items):
   - Filter order_details['query_items'] to ONLY items where found=True
   - Store filtered list as available_items
   - Keep unavailable_items list from step 2
   - Proceed to Phase 1 using ONLY available_items
   - Include unavailable_items list in your final response
   - MUST mention unavailable items in quotation_text

UNDER NO CIRCUMSTANCES should you calculate quotes for items with found=False.

IMPORTANT:
If no items in inventory can be ordered (i.e., all items have 'found' = False/false or item_name = 'N/A'/'Not Found'/'None') OR inventory_items is empty:
  - Return quotation_items as empty list   
  - Set subtotal, discount_percent, discount_amount, final_total to 0.0
  - Provide a polite message in quotation_text indicating no items could be ordered and no quote was generated
  - Do NOT call any tools and give a response with empty quotation_items and zero totals
  
## CRITICAL: TOOL EXECUTION WORKFLOW

⚠️ **YOU MUST FOLLOW THIS EXACT SEQUENCE** ⚠️

PHASE 1: CALCULATION (Use Tools)
DO NOT return any answer during this phase. Only call tools.

Step 1: Check past quotes
- Call find_similar_past_quotes with list of query_items[query] from query_items in order_details
- If past quotes found, use them to guide quote style, tone, and wording.

Step 2: Calculate individual item quotes
- For EACH item in order_details, call calculate_single_item_quote
- Collect all item results

Step 3: Calculate order total
- Call calculate_order_total with all items
- Get total_quantity and subtotal

Step 4: Calculate bulk discount
- Call calculate_bulk_discount(total_quantity, order_size)
- Get discount_percent

Step 5: Calculate final amounts
- discount_amount = subtotal × (discount_percent / 100)
- final_total = subtotal - discount_amount

PHASE 2: RESPONSE (Return Final Answer)
Only AFTER completing ALL tool calls in Phase 1, return your final answer as a Python dictionary.

⚠️ **DO NOT CALL ANY TOOLS DURING PHASE 2** ⚠️
⚠️ **DO NOT RETURN AN ANSWER DURING PHASE 1** ⚠️

---

## Available Tools

### 1. calculate_single_item_quote
Calculates quotes for single item with automatic markup and discounts.

**Input Format (dictionary):**
{
    "item_name": "Glossy paper",
    "order_quantity": 200,
    "unit_price": 0.2
}

**Returns:**
{
    "success": true,
    "item_name": "Glossy paper",
    "order_quantity": 200,
    "cost_price": 0.2,  # Original unit_price (cost)
    "selling_price": 0.24,  # After 20% markup
    "markup_percent": 20.0,
    "subtotal": 40.0,
    "markup_amount": 8.0,
    "total_amount": 48.0
}

**IMPORTANT: When building quotation_items, you MUST include:**
- unit_price: Use "cost_price" from tool result
- selling_price: Use "selling_price" from tool result

### 2. calculate_order_total
Calculates total of the quote for multiple items.

**Input Format:**
{
    "quotation_items": [item1_dict, item2_dict, ...]
}

**Returns:**
{
    "success": true,
    "items_count": 3,
    "total_quantity": 800,
    "total_amount": 180.0
}

### 3. calculate_bulk_discount
Calculates bulk discount percentage.

**Input:**
- total_quantity (int): Total units across all items
- need_size (str): Order size ("SMALL", "MEDIUM", "LARGE")

**Returns:**
- float: Discount percentage (e.g., 10.0 for 10%)


### 4. find_similar_past_quotes
Finds similar past quotes based on search terms to use as a basis for the quote wording.

**Input:**  
- search_terms (list of str): Keywords to search for in past quotes

**Returns:**    
- List of past quotes (list of dictionaries)    

---

## Execution Example

Given order with 2 items:
1. Call calculate_single_item_quote for item 1 → get result1
2. Call calculate_single_item_quote for item 2 → get result2
3. Call calculate_order_total([result1, result2]) → get total_quantity, subtotal
4. Call calculate_bulk_discount(total_quantity, order_size) → get discount_percent
5. Calculate: discount_amount = subtotal × (discount_percent / 100)
6. Calculate: final_total = subtotal - discount_amount
7. NOW return final answer with all results

---

## Response Format (Phase 2 Only)

Return a Python dictionary (NOT JSON string, NO markdown):

{
    "quotation_text": "Thank you for your order! ...",
    "order_id": "ORD_12345678",
    "items_count": 2,
    "quotation_items": [
        {
            "item_name": "Photo paper",
            "category": "paper",
            "order_quantity": 300,
            "unit_price": 0.25,  # REQUIRED: Original cost price
            "selling_price": 0.30,  # REQUIRED: Price with 20% markup
            "total": 90.00
        }
    ],
    "unavailable_items": [  # NEW: Items that cannot be ordered
        {
            "query": "balloons",
            "order_quantity": 200,
            "reason": "Not available in catalog"
        }
    ],
    "subtotal": 180.00,
    "discount_percent": 5.0,
    "discount_amount": 9.00,
    "final_total": 171.00
}

**CRITICAL: Each item in quotation_items MUST include BOTH:**
- **unit_price**: Original cost price from inventory
- **selling_price**: Price after 20% markup (unit_price × 1.2)

**CRITICAL: UNAVAILABLE ITEMS HANDLING:**

You MUST always check for and include unavailable items in your response!

- Identify ALL items where found=False or item_name='N/A'/'Not Found'/'None'
- Add these to "unavailable_items" list with:
  * query (original search term from order_details)
  * order_quantity (requested quantity)
  * reason (message explaining why not available)
- In quotation_text, politely mention unavailable items at the END after pricing
- Suggest alternatives or express willingness to help find substitutes
- DO NOT include unavailable items in quotation_items or pricing calculations

Example unavailable_items format:
[
    {
        "query": "posters",
        "order_quantity": 2000,
        "reason": "Similar item NOT found in catalog, item cannot be ordered"
    },
    {
        "query": "tickets",
        "order_quantity": 10000,
        "reason": "Similar item NOT found in catalog, item cannot be ordered"
    }
]

---

## Quote Response Guidelines

### When to Mention Discounts
- **Discount applies (discount_percent > 0)**: 
  "Since you're ordering X units, I'm pleased to apply a Y% bulk discount!"
  
- **No discount (discount_percent = 0)**:
  Simply state the total without mentioning discounts

### Professional Tone Examples

**With discount:**
"Thank you for your order! For your upcoming festival, here's your quote:

- 500 sheets of A4 paper at $0.06/sheet = $30.00
- 300 sheets of cardstock at $0.18/sheet = $54.00

Subtotal: $84.00

Since you're ordering 800 total units, I'm pleased to apply a 5% bulk discount!
Discount: -$4.20

Your total: $79.80"

**No discount:**
"Thank you for your order! For your office supplies, here's your quote:

- 100 sheets of A4 paper at $0.06/sheet = $6.00
- 50 sheets of photo paper at $0.30/sheet = $15.00

Your total: $21.00"

**With unavailable items:**
"Thank you for your order! For your upcoming concert, we have calculated the costs for 5,000 flyers at $0.18 each. Since you're ordering in bulk, I'm pleased to apply a 5% discount to your order.

Subtotal: $900.00
Discount (5%): -$37.50
Your total: $862.50

**IMPORTANT - ITEMS NOT AVAILABLE:**
We apologize, but the following items from your request are not currently available in our catalog:

• 2,000 posters - Similar item NOT found in catalog, item cannot be ordered
• 10,000 tickets - Similar item NOT found in catalog, item cannot be ordered

We specialize in paper supplies and related products. If you need alternative products or would like recommendations for similar items we can provide, please contact us at orders@munderdifflin.com and we'll be happy to help find solutions for your event!"

---

## Pricing Categories Reference

### Paper Category (per sheet)
Examples: A4 paper, cardstock, photo paper, glossy paper
- Quantities counted per sheet
- 1 ream = 500 sheets

### Product Category (per unit)
Examples: paper plates, cups, napkins, envelopes, notepads
- Quantities counted per item

### Large Format Category (per unit)
Examples: poster paper (24x36"), banner rolls (36" width)
- Quantities counted per piece

### Specialty Category (per unit)
Examples: 100lb cover stock, 80lb text paper, specialty cardstock
- Quantities counted per sheet

---

## CRITICAL REMINDERS

1. ✅ Call ALL tools in Phase 1 BEFORE returning answer
2. ✅ Use calculate_single_item_quote for EACH item
3. ✅ Use calculate_order_total to sum all items
4. ✅ Use calculate_bulk_discount for discount percentage
5. ✅ Return Python dict (not JSON string)
6. ✅ Only mention discount if discount_amount > 0
7. ✅ ALWAYS include BOTH unit_price AND selling_price in each quotation_item
8. ❌ DO NOT return answer while calling tools
9. ❌ DO NOT call tools after starting to return answer
10. ❌ DO NOT skip any calculation steps
11. ❌ DO NOT omit unit_price or selling_price fields
12. ✅ ALWAYS include unavailable_items list in response (even if empty)
13. ✅ ALWAYS mention unavailable items in quotation_text if any exist



Remember: 
1. TOOLS FIRST, ANSWER LAST!
2. ALWAYS include unavailable_items field in your response (even if empty list [])
3. ALWAYS mention unavailable items in quotation_text if any exist
4. Extract unavailable items BEFORE starting Phase 1 calculations
""",
        )

    def run(
        self,
        query: str,
        order_details: Dict[str, Any],
        inventory_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a customer query related to pricing and quotes.

        Args:
            query (str): Customer's pricing-related question or request.
            order_details (Dict[str, Any]): Details of the order.
            inventory_details (Dict[str, Any]): Details of the inventory status.

        Returns:
            Dict[str, Any]: Pricing Agent's response after processing the query.
        """
        return super().run(
            query,
            additional_args={
                "order_details": order_details,
                "inventory_details": inventory_details,
            },
        )


class OrderingAgent(ToolCallingAgent):
    """
    Order fulfillment agent that processes multi-item orders and updates inventory.

    Responsibilities:
    - Validate orders against current inventory
    - Trigger reorders when needed
    - Process confirmed orders through the fulfillment system
    - Update inventory levels after fulfillment

    """

    def __init__(self, model: Any):
        """
        Initialize the Fulfillment Agent.

        Args:
            model: LLM model instance
        """
        super().__init__(
            tools=[
                fulfill_single_item_order,
                # reorder_item_tool,
                # generate_tracking_number,
            ],
            model=model,
            name="ordering_agent",
            description="""
Order Fulfillment Agent for Munder Difflin Paper Supply

RESPONSIBILITIES:
- Fulfill multi-item orders and update inventory
- Process each item with complete fulfillment details
- Generate tracking numbers for shipped orders
- Aggregate fulfillment results into comprehensive order confirmation

WORKFLOW:

STEP 1: PROCESS EACH ITEM
For each item in quote_details['quotation_items'] where order_quantity > 0:
  
  a) Extract from quote_details['quotation_items'][i]:
     - item_name
     - category
     - order_quantity
     - unit_price (cost price)
     - selling_price (price to customer)
  
  b) Find matching item in inventory_details['inventory_items'] by item_name:
     - reorder_quantity (from matched inventory item)
  
  c) Get request_date from additional_args
  
  d) Call fulfill_single_item_order with ALL 7 parameters:
     fulfill_single_item_order(
         item_name=item_name,
         category=category,
         order_quantity=order_quantity,
         unit_price=unit_price,
         selling_price=selling_price,
         request_date=request_date,
         reorder_quantity=reorder_quantity
     )

STEP 2: GENERATE TRACKING
After ALL items are fulfilled, call generate_tracking_number() once.

STEP 3: BUILD RESPONSE
Aggregate all fulfillment results with tracking number.

DATA STRUCTURE REFERENCE:

quote_details structure:
{
    "quotation_text": str,
    "order_id": str,
    "quotation_items": [
        {
            "item_name": "Photo paper",
            "category": "paper",
            "order_quantity": 300,
            "unit_price": 0.25,
            "selling_price": 0.30,
            "total": 90.00
        },
        ...
    ],
    "subtotal": float,
    "discount_percent": float,
    "final_total": float
}

inventory_details structure:
{
    "inventory_items": [
        {
            "item_name": "Photo paper",
            "order_quantity": 300,
            "current_stock": 788,
            "reorder_quantity": 0,
            "min_stock_level": 143,
            "estimated_delivery_date": "2025-04-01"
        },
        ...
    ],
    "delivery_date_comparison": {...}
}

TOOLS AVAILABLE:

1. fulfill_single_item_order(item_name, category, order_quantity, unit_price, selling_price, request_date, reorder_quantity)
   - Processes order fulfillment for ONE item
   - Updates inventory by reducing stock
   - Records sales transaction
   - Returns: fulfillment details including new_stock, fulfillment_status
   - Call ONCE per item with order_quantity > 0

2. generate_tracking_number()
   - Generates unique tracking number for shipment
   - Returns: tracking number string (e.g., "TRK1A2B3C4D5E6F")
   - Call ONCE after all items are fulfilled

CRITICAL RULES:
- You MUST call fulfill_single_item_order for EVERY item with order_quantity > 0
- You MUST pass ALL 7 parameters to fulfill_single_item_order
- You MUST match items between quote_details and inventory_details by item_name
- You MUST call generate_tracking_number exactly once after all fulfillments
- If reorder_quantity > 0 for an item, pass it to fulfill_single_item_order

OUTPUT FORMAT:
{
    "order_id": str,
    "tracking_number": str,
    "order_delivery_date": str,  # Latest estimated_delivery_date from inventory_details
    "total_amount": float,  # From quote_details['final_total']
    "items": [
        {
            "item_name": str,
            "unit_price": float,
            "selling_price": float,
            "total": float,
            "estimated_delivery_date": str,
            "order_quantity": int,
            "current_stock": int,
            "reorder_quantity": int,
            "new_stock": int,
            "fulfillment_status": "Completed" | "Failed"
        },
        ...
    ]
}

EXAMPLE ITERATION:
Item: "Photo paper"
- From quote_details: item_name="Photo paper", category="paper", order_quantity=300, 
  unit_price=0.25, selling_price=0.30
- From inventory_details (match by item_name): reorder_quantity=0, estimated_delivery_date="2025-04-01"
- From additional_args: request_date="2025-03-28"
- Call: fulfill_single_item_order("Photo paper", "paper", 300, 0.25, 0.30, "2025-03-28", 0)

IMPORTANT:
If no items found are found in inventory (all found=False/false):
 - Return items as empty list
 - Set total_amount to 0.0
 - Do not call fulfill_single_item_order or generate_tracking_number

Ensure every item in quote_details['quotation_items'] is processed according to this workflow.
""",
        )

    def run(
        self,
        query: str,
        quote_details: Dict[str, Any],
        inventory_details: Dict[str, Any],
        request_date: str,
    ) -> str:
        """
        Process the customer order fulfillment.

        Args:
            query (str): Customer's order fulfillment request.
            quote_details (Dict[str, Any]): Details of the quote.
            inventory_details (Dict[str, Any]): Details of the inventory status.
            request_date (str): Date of the order request.

        Returns:
            Dict[str, Any]: Pricing Agent's response after processing the query.
        """
        return super().run(
            query,
            additional_args={
                "quote_details": quote_details,
                "inventory_details": inventory_details,
                "request_date": request_date,
            },
        )


class OrchestratorAgent:
    """
    Master orchestrator agent that coordinates all specialized agents.

    Responsibilities:
    - Route tasks to appropriate specialized agents
    - Coordinate multi-step workflows
    - Synthesize responses from multiple agents
    - Provide coherent customer-facing responses
    """

    def __init__(self, model):

        self.orderprocessing_agent = OrderProcessingAgent(model)
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.ordering_agent = OrderingAgent(model)

        print("Orchestrator Agent initialized with inheritance-based agents")

    def process_query(self, customer_query: str) -> str:
        """
        Process a customer quotation query end-to-end.
        Args:
            customer_query (str): Customer's quotation request.

        Returns:
            str: Agent's response after processing the query.
        """

        try:

            order_id = generate_order_id()
            state["order_id"] = order_id
            # Step 1: Parse the order details from the customer query
            example = {
                "request_date": "2024-09-14",
                "expected_delivery_date": "2024-09-15",
                "order_id": "ORD_12345678",
                "customer_query": " I want to order 500 sheets of recycled A4 cardstock, 300 sheets of A3 photo paper in assorted colors and 200 balloons.",
                "query_items": [
                    {
                        "query": "cardstock",
                        "item_name": "Cardstock",
                        "order_quantity": 500,
                        "category": "paper",
                        "size": "A4",
                        "notes": "recycled",
                        "unit_price": 0.15,
                        "source": "catalog",
                        "found": True,
                        "message": "Found similar item in catalog, item can be ordered",
                    },
                    {
                        "query": "photo paper",
                        "item_name": "Photo paper",
                        "order_quantity": 300,
                        "category": "paper",
                        "size": "A3",
                        "notes": "assorted colors",
                        "unit_price": 0.25,
                        "source": "catalog",
                        "found": True,
                        "message": "Found similar item in catalog, item can be ordered",
                    },
                    {
                        "query": "balloons",
                        "item_name": None,
                        "order_quantity": 200,
                        "category": None,
                        "size": None,
                        "notes": None,
                        "unit_price": None,
                        "source": "catalog",
                        "found": False,
                        "message": "Item not found in catalog, item cannot be ordered",
                    },
                ],
            }
            prompt = f"""
    Customer Query: {customer_query}
    Order ID: {order_id}

    Parse the order details from the customer query

    Return details of the customer query(request_date, expected_delivery_date, order_id, customer_query) including a list of query items with their the search query term, closest matches of item name, order quantities, categories, sizes, additional notes, unit prices, sources, and whether the item was found. Return your response as a JSON string.

    Always convert reams to sheets when the customer specifies quantity in reams (1 ream = 500 sheets).

    Example format for response:
    {json.dumps(example, indent=4)}

    {JSON_RESPONSE_INSTRUCTIONS}
            """
            order_details_response = self.orderprocessing_agent.run(prompt)
            # Parse JSON response to dictionary
            order_details_response = normalize_agent_response(order_details_response)
            state["order_details_response"] = order_details_response

            # self._dump_state("ORDER DETAILS COMPLETED")

            inventory_example = {
                "inventory_items": [
                    {
                        "item_name": "Glossy paper",
                        "order_quantity": 200,
                        "current_stock": 587,
                        "reorder_quantity": 0,
                        "min_stock_level": 147,
                        "estimated_delivery_date": "2025-04-02",
                    },
                    {
                        "item_name": "Cardstock",
                        "order_quantity": 100,
                        "current_stock": 595,
                        "reorder_quantity": 0,
                        "min_stock_level": 148,
                        "estimated_delivery_date": "2025-04-03",
                    },
                    {
                        "item_name": "Colored paper",
                        "order_quantity": 100,
                        "current_stock": 788,
                        "reorder_quantity": 0,
                        "min_stock_level": 143,
                        "estimated_delivery_date": "2025-04-01",
                    },
                ],
                "delivery_date_comparison": {
                    "requested_delivery_date": "2025-04-05",
                    "estimated_delivery_date": "2025-04-03",
                    "is_feasible": True,
                    "message": "The estimated delivery date meets the requested delivery date.",
                },
                "inventory_snapshot": {
                    "snapshot_date": "2025-03-28",
                    "total_items": 1500,
                },
            }
            inventory_prompt = f"""
    Process the inventory for this order and check delivery feasibility.

    ORDER DETAILS PROVIDED:
    {json.dumps(order_details_response, indent=2)}

    INSTRUCTIONS:
    1. For each item in query_items where found=True:
    - Call check_current_stock to get current stock levels
    - Call calculate_reorder_quantity to determine reorder needs
    - Call check_delivery_timeline to get estimated delivery date

    2. After processing ALL items:
    - Identify the LATEST estimated_delivery_date from all items
    - Call compare_delivery_dates using:
        * expected_delivery_date from order_details_response['expected_delivery_date']
        * Latest estimated_delivery_date from the items you processed

    3. Return a JSON string with THREE keys:
    a) "inventory_items": list of all processed items with stock info
    b) "delivery_date_comparison": result from compare_delivery_dates tool
    c) "inventory_snapshot": current inventory status including low stock and out of stock items
    4. Follow the EXACT output format shown in this example:
    {json.dumps(inventory_example, indent=2)}

    IMPORTANT:
    CRITICAL: You MUST call compare_delivery_dates once after processing all items.

    Expected format for response (return as JSON string):
    {json.dumps(inventory_example, indent=2)}

    {JSON_RESPONSE_INSTRUCTIONS}
    """
            inventory_response = self.inventory_agent.run(
                inventory_prompt, order_details_response
            )
            # Parse JSON response to dictionary
            inventory_response = normalize_agent_response(inventory_response)
            state["inventory_response"] = inventory_response

            # self._dump_state("INVENTORY COMPLETED")

            quoting_example = {
                "quotation_text": "Thank you for your order! For your upcoming concert, we have calculated the costs for 5,000 flyers at $0.18 each. Since you're ordering in bulk, I'm pleased to apply a 5% discount to your order.\n\nSubtotal: $900.00\nDiscount (5%): -$37.50\nYour total: $862.50\n\n**IMPORTANT - ITEMS NOT AVAILABLE:**\nWe apologize, but the following items from your request are not currently available in our catalog:\n\n• 2,000 posters - Similar item NOT found in catalog\n• 10,000 tickets - Similar item NOT found in catalog\n\nWe specialize in paper supplies. If you need alternatives, please contact us at orders@munderdifflin.com!",
                "order_id": "ORD_12345678",
                "items_count": 1,
                "quotation_items": [
                {
                "item_name": "Flyers",
                "category": "product",
                "order_quantity": 5000,
                "unit_price": 0.15,  # Cost price from inventory
                "selling_price": 0.18,  # unit_price * 1.2 markup
                "total": 900.00,
                }
                ],
                "unavailable_items": [  # CRITICAL: ALWAYS include this field
                {
                "query": "posters",
                "order_quantity": 2000,
                "reason": "Similar item NOT found in catalog, item cannot be ordered"
                },
                {
                    "query": "tickets",
                "order_quantity": 10000,
                "reason": "Similar item NOT found in catalog, item cannot be ordered"
                }
                ],
                "subtotal": 900.00,
                "discount_percent": 5.0,
                "discount_amount": 37.50,
                "final_total": 862.50,
            }
            quoting_prompt = f"""
    Generate a complete pricing quote for this order.

    ORDER DETAILS:
    {json.dumps(order_details_response, indent=2)}

    INVENTORY DETAILS:
    {json.dumps(inventory_response, indent=2)}

    ---

    EXECUTION INSTRUCTIONS:

    ⚠️ PHASE 1: CALCULATIONS (DO NOT RETURN ANSWER YET)

    Step 1: Check for past quotes
    - Call find_similar_past_quotes with:
    * search_terms: list of item_name from order_details['query_items'] where found=True
    - If past quotes found, use them to inform pricing strategy decisions

    Step 2: For EACH item in order_details['query_items'] where found=True:
    - Call calculate_single_item_quote with:
        * item_name
        * order_quantity
        * unit_price
    - Store the result

    Step 3: After ALL items are calculated:
    - Call calculate_order_total with list of all item results
    - This gives you: total_quantity, subtotal

    Step 4: Calculate discount:
    - Call calculate_bulk_discount(total_quantity, order_size)
    - This gives you: discount_percent

    Step 5: Calculate final amounts:
    - discount_amount = subtotal × (discount_percent / 100)
    - final_total = subtotal - discount_amount

    ⚠️ PHASE 2: RETURN ANSWER (AFTER ALL CALCULATIONS)

    Only after completing ALL steps above, return your final answer as a JSON string.
    
    ⚠️ CRITICAL: Your response MUST include the "unavailable_items" field!
    - If there are unavailable items (found=False), include them in the list
    - If ALL items are available, include unavailable_items: []
    - DO NOT omit this field

    ---

    REQUIRED OUTPUT FORMAT:

    JSON string with these fields:
    - quotation_text: str (customer-friendly quote explanation - MUST mention unavailable items if any exist)
    - order_id: str (from order_details)
    - items_count: int (number of items quoted)
    - quotation_items: list[dict] (all items with pricing details)
    **EACH ITEM MUST HAVE:**
    - item_name, category, order_quantity
    - unit_price (original cost price)
    - selling_price (price after 20% markup)
    - total (selling_price × quantity)
    - unavailable_items: list[dict] (CRITICAL: ALWAYS include this field, even if empty [])
    **EACH UNAVAILABLE ITEM MUST HAVE:**
    - query: str (original search term from order_details)
    - order_quantity: int (requested quantity)
    - reason: str (why item cannot be ordered)
    - subtotal: float (before discount)
    - discount_percent: float (0.0 if no discount)
    - discount_amount: float (0.0 if no discount)
    - final_total: float (subtotal - discount_amount)

    If discount_percent = 0:
    - Set discount_amount = 0.0
    - Set final_total = subtotal
    - Do NOT mention discount in quotation_text

    If discount_percent > 0:
    - Calculate discount_amount
    - Calculate final_total
    - DO mention discount in quotation_text

    IMPORTANT:
    If no items found (all found=False/false):
    - Return quotation_items as empty list   
    - Set subtotal, discount_percent, discount_amount, final_total to 0.0
    - Provide a polite message in quotation_text indicating no items could be ordered and no quote was generated
    ---

    EXAMPLE EXPECTED OUTPUT:

    {json.dumps(quoting_example, indent=2)}

    ---

    CRITICAL RULES:
    1. Complete ALL tool calls BEFORE returning answer
    2. Do NOT return answer until ALL calculations are done
    3. Return as JSON string (no markdown code blocks)
    4. Follow the exact output format shown above

    BEGIN PHASE 1 NOW - Calculate all pricing using tools.

    {JSON_RESPONSE_INSTRUCTIONS}
    """

            quoting_response = self.quoting_agent.run(
                quoting_prompt,
                order_details=order_details_response,
                inventory_details=inventory_response,
            )
            # Parse JSON response to dictionary
            quoting_response = normalize_agent_response(quoting_response)
            state["quoting_response"] = quoting_response

            # self._dump_state("PRICING COMPLETED")

            fulfillment_example = {
                "order_id": "ORD_12345678",
                "tracking_number": "None",
                "order_delivery_date": "2025-04-03",
                "total_amount": 243.00,
                "items": [
                    {
                        "item_name": "Photo paper",
                        "unit_price": 0.25,
                        "selling_price": 0.30,
                        "total": 75.00,
                        "estimated_delivery_date": "2025-04-01",
                        "order_quantity": 100,
                        "current_stock": 788,
                        "reorder_quantity": 0,
                        "min_stock_level": 143,
                        "new_stock": 688,
                        "fulfillment_status": "Completed",
                    },
                    {
                        "item_name": "Cardstock",
                        "unit_price": 0.2,
                        "total": 100.00,
                        "estimated_delivery_date": "2025-04-02",
                        "order_quantity": 50,
                        "current_stock": 120,
                        "reorder_quantity": 40,
                        "min_stock_level": 110,
                        "new_stock": 110,
                        "fulfillment_status": "Completed",
                    },
                ],
            }

            # Step 5: Fulfill the order
            fullfillment_prompt = f"""
    Process order fulfillment for all items in the quote.

    OUTPUT FORMAT REQUIREMENTS:
    - You MUST return a JSON string (not a Python dictionary)
    - Do NOT wrap your response in markdown code blocks
    - Do NOT include any text before or after the JSON string
    - Return ONLY the JSON string

    DATA PROVIDED:

    QUOTE DETAILS:
    {json.dumps(quoting_response, indent=2)}

    INVENTORY DETAILS:
    {json.dumps(inventory_response, indent=2)}

    REQUEST DATE: {state["request_date"]}

    INSTRUCTIONS:

    1. For EACH item in quote_details['quotation_items'] where order_quantity > 0:
    
    a) Extract from quote_details['quotation_items']:
        - item_name, category, order_quantity, unit_price, selling_price, total
    
    b) Find matching item in inventory_details['inventory_items'] by item_name:
        - reorder_quantity, estimated_delivery_date, current_stock
    
    c) Call fulfill_single_item_order with all 7 parameters:
        - item_name (from quote)
        - category (from quote)
        - order_quantity (from quote)
        - unit_price (from quote)
        - selling_price (from quote)
        - request_date (provided above)
        - reorder_quantity (from inventory)

    2. After ALL items are fulfilled:
    - Call generate_tracking_number() to get tracking number

    3. Build response with:
    - order_id: from quote_details['order_id']
    - tracking_number: from generate_tracking_number()
    - order_delivery_date: LATEST estimated_delivery_date from all items
    - total_amount: from quote_details['final_total']
    - items: list combining fulfillment results with inventory details

    CRITICAL: You MUST call fulfill_single_item_order for every item, passing ALL 7 parameters.

    Expected response format (return as JSON string):
    {json.dumps(fulfillment_example, indent=2)}

    {JSON_RESPONSE_INSTRUCTIONS}
    """
            fulfillment_response = self.ordering_agent.run(
                fullfillment_prompt,
                quote_details=quoting_response,
                inventory_details=inventory_response,
                request_date=state["request_date"],
            )
            # Parse JSON response to dictionary
            fulfillment_response = normalize_agent_response(fulfillment_response)
            state["fulfillment_response"] = fulfillment_response

            # self._dump_state("ORDER FULFILLMENT COMPLETED")
            # Step 6: Build the final response to the customer

            # state = normalize_agent_response(state)

            # build a list of quotation items for the response
            quotation_items = []
            quotation_response_items = state["quoting_response"]["quotation_items"]
            for item in quotation_response_items:
                # Handle both selling_price and unit_price fields
                price = item.get("selling_price")
                quotation_items.append(
                    {
                        "item_name": item["item_name"],
                        "order_quantity": item["order_quantity"],
                        "price": price,
                        "total": item["total"],
                    }
                )

            # Craft the reponse to the customer with all the details using details from the global state
            response = {
                "request_date": state["request_date"],
                "original_request": state["original_request"],
                "customer_job": state["customer_job"],
                "event_type": state["event_type"],
                "need_size": state["need_size"],
                "order_id": state["order_id"],
                "quotation": state["quoting_response"]["quotation_text"],
                "quotation_items": quotation_items,
                "unavailable_items": state["quoting_response"].get("unavailable_items", []),
                "order_delivery_date": state["fulfillment_response"][
                    "order_delivery_date"
                ],
                "total_amount": state["fulfillment_response"]["total_amount"],
                "tracking_number": state["fulfillment_response"]["tracking_number"],
            }

            formatted_response = format_order_confirmation_email(response)

            return formatted_response
        except Exception as e:
            print(f"Error occurred while formatting response: {e}")

    def _dump_state(self, step_name=""):
        """Dump the current state."""
        print(f"\n===== {step_name} STATE =====")
        print(json.dumps(state, indent=4))


# initialize global quotation state
global quotation_state
# quotation_state = QuoteRequestState(customer_job="", event_type="", order_size=OrderSize.SMALL, items=[])


# initialize global state
state = {}


def dump_state():
    """Dump the current state."""
    print("\n===== GLOBAL STATE =====")
    print(json.dumps(state, indent=2))


# Initialize semantic search
search_engine = None  #


# Test scenarios
def run_test_scenarios():
    """Run test scenarios."""
    print("Initializing Database...")
    init_database(db_engine)
    search_engine = initialize_semantic_search(db_engine, paper_supplies=paper_supplies)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    print("Initializing Multi-Agent System...")
    model = get_llm_model()

    orchestrator = OrchestratorAgent(model)
    print("Multi-Agent System Ready!\n")

    results = []
    for idx, row in quote_requests_sample.iterrows():

        request_date = row["request_date"].strftime("%Y-%m-%d")

        match row["need_size"].lower():
            case "small":
                order_size = OrderSize.SMALL
            case "medium":
                order_size = OrderSize.MEDIUM
            case "large":
                order_size = OrderSize.LARGE
            case _:
                order_size = OrderSize.SMALL

        print(f"\n===== Processing Request {idx+1} =====")
        print(f"Original Request: {row['request']}")

        state["request_date"] = request_date
        state["original_request"] = row["request"]
        state["customer_job"] = row["job"]
        state["event_type"] = row["event"]
        state["need_size"] = order_size

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = (
            f"{remove_quotes(row['request'])} (Date of request: {request_date})"
        )
        response = orchestrator.process_query(request_with_date)

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        state.clear()
        time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results_v1.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
