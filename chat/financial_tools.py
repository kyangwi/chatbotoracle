"""
chat/financial_tools.py
-----------------------
Financial analytics tools & calculators for NSSF Uganda General Ledger.
Enables precise rate-of-change, CAGR, exchange rate variance, and arithmetic calculations,
as well as programmatic data retrieval for comparative period analysis.
"""

import ast
import math
import operator
import re
from typing import Optional, Dict, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Safe Financial Math Engine
# ---------------------------------------------------------------------------
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op = type(node.op)
        if op in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op](left, right)
        raise ValueError(f"Unsupported binary operator: {op}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        op = type(node.op)
        if op in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op](operand)
        raise ValueError(f"Unsupported unary operator: {op}")
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        args = [_safe_eval_node(arg) for arg in node.args]
        if func_name == "abs":
            return abs(args[0])
        elif func_name == "round":
            return round(args[0], int(args[1]) if len(args) > 1 else 2)
        elif func_name == "sqrt":
            return math.sqrt(args[0])
        elif func_name == "min":
            return min(args)
        elif func_name == "max":
            return max(args)
        raise ValueError(f"Unsupported function call: {func_name}")
    else:
        raise ValueError(f"Unsupported expression element: {type(node)}")


# ---------------------------------------------------------------------------
# Tool 1: Financial Calculator
# ---------------------------------------------------------------------------
class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Mathematical expression to evaluate, e.g. '(250000000 - 180000000) / 180000000 * 100'"
    )


@tool("financial_calculator", args_schema=CalculatorInput)
def financial_calculator(expression: str) -> str:
    """
    Safely evaluates mathematical and financial formulas with high precision.
    Supports addition, subtraction, multiplication, division, powers (**), abs(), and round().
    Use this tool whenever exact financial arithmetic or variance calculation is needed.
    """
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", expression.strip())
    try:
        parsed = ast.parse(cleaned, mode="eval")
        result = _safe_eval_node(parsed.body)
        if isinstance(result, float):
            formatted = f"{result:,.4f}".rstrip("0").rstrip(".")
        else:
            formatted = f"{result:,}"
        return f"Result of `{expression}` = {formatted}"
    except Exception as e:
        return f"Calculation error for `{expression}`: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Rate of Change Calculator
# ---------------------------------------------------------------------------
class RateOfChangeInput(BaseModel):
    old_value: float = Field(description="The baseline or prior period financial value")
    new_value: float = Field(description="The comparison or current period financial value")
    label: str = Field(
        default="",
        description="Optional name or label for the metric (e.g. 'Operating Expenses', 'USD Rate')",
    )


@tool("calculate_rate_of_change", args_schema=RateOfChangeInput)
def calculate_rate_of_change(old_value: float, new_value: float, label: str = "") -> str:
    """
    Calculates the exact percentage rate of change and absolute difference between two values.
    Formula: Rate of Change (%) = ((new_value - old_value) / |old_value|) * 100.
    Provides direction (increase/decrease/unchanged), multiplier, and formatted figures.
    """
    if old_value == 0:
        diff = new_value
        return (
            f"{label + ': ' if label else ''}"
            f"Baseline value is 0.00. Absolute change is {diff:+,.2f}. "
            f"Percentage rate of change is mathematically undefined (division by zero)."
        )

    diff = new_value - old_value
    pct = (diff / abs(old_value)) * 100.0
    direction = "Increase" if diff > 0 else ("Decrease" if diff < 0 else "Unchanged")
    sign = "+" if diff > 0 else ""
    ratio = new_value / old_value if old_value != 0 else 0

    return (
        f"{label + ': ' if label else ''}"
        f"{sign}{pct:.2f}% {direction} "
        f"(Absolute Change: {diff:+,.2f}, from {old_value:,.2f} to {new_value:,.2f}, "
        f"ratio: {ratio:.4f}x)"
    )


# ---------------------------------------------------------------------------
# Tool 3: Compound Annual Growth Rate (CAGR) Calculator
# ---------------------------------------------------------------------------
class CAGRInput(BaseModel):
    start_value: float = Field(description="Initial value at the beginning of the multi-period span")
    end_value: float = Field(description="Final value at the end of the span")
    periods: float = Field(description="Number of years or compounding periods between start and end")
    label: str = Field(default="", description="Optional metric description")


@tool("calculate_cagr", args_schema=CAGRInput)
def calculate_cagr(start_value: float, end_value: float, periods: float, label: str = "") -> str:
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    Formula: CAGR = (end_value / start_value) ** (1 / periods) - 1.
    Useful for multi-year trends in ledger transactions, fund assets, or recurring expenditures.
    """
    if periods <= 0:
        return "Periods must be greater than 0 to compute CAGR."
    if start_value <= 0 or end_value <= 0:
        return "CAGR requires positive start and end values."

    cagr = ((end_value / start_value) ** (1.0 / periods) - 1.0) * 100.0
    sign = "+" if cagr > 0 else ""
    return (
        f"{label + ': ' if label else ''}"
        f"CAGR over {periods} periods is {sign}{cagr:.2f}% per period "
        f"(Start: {start_value:,.2f}, End: {end_value:,.2f})"
    )


# ---------------------------------------------------------------------------
# Tool 4: Currency Exchange Rate Change & FX Impact Calculator
# ---------------------------------------------------------------------------
class CurrencyRateChangeInput(BaseModel):
    amount: float = Field(description="Transaction amount in foreign or base currency")
    from_currency: str = Field(description="Source currency code (e.g. 'USD', 'EUR', 'GBP', 'UGX')")
    to_currency: str = Field(default="UGX", description="Target currency code (defaults to 'UGX')")
    historical_rate: Optional[float] = Field(
        default=None,
        description="Historical exchange rate recorded at transaction time (glcrr/glhcrr)",
    )
    current_rate: Optional[float] = Field(
        default=None,
        description="Current/comparison exchange rate (if omitted, uses default market rates)",
    )


@tool("calculate_currency_rate_change", args_schema=CurrencyRateChangeInput)
def calculate_currency_rate_change(
    amount: float,
    from_currency: str,
    to_currency: str = "UGX",
    historical_rate: Optional[float] = None,
    current_rate: Optional[float] = None,
) -> str:
    """
    Calculates exchange rate movement, percentage variation, and the resulting foreign exchange
    variance / gain / loss on monetary amounts in the General Ledger.
    """
    from_c = from_currency.upper().strip()
    to_c = to_currency.upper().strip()

    rate_table = {
        "USD": 3720.0,
        "EUR": 4050.0,
        "GBP": 4750.0,
        "KES": 28.8,
        "RWF": 2.7,
        "UGX": 1.0,
        "SHS": 1.0,
    }

    h_rate = historical_rate
    c_rate = current_rate or rate_table.get(from_c, 3700.0)

    if h_rate is None:
        return (
            f"Converting {amount:,.2f} {from_c} to {to_c} at rate {c_rate:,.2f}: "
            f"Result = {amount * c_rate:,.2f} {to_c}."
        )

    rate_diff = c_rate - h_rate
    rate_pct = (rate_diff / h_rate) * 100.0 if h_rate != 0 else 0.0

    orig_converted = amount * h_rate
    new_converted = amount * c_rate
    fx_variance = new_converted - orig_converted

    gain_loss = "FX Loss (expense increased)" if fx_variance > 0 else "FX Gain (savings/favorable)"
    sign = "+" if rate_diff > 0 else ""

    lines = [
        f"Exchange Rate Change ({from_c}/{to_c}):",
        f"- Historical Rate: {h_rate:,.2f} | Current Rate: {c_rate:,.2f} ({sign}{rate_pct:.2f}% rate movement)",
        f"- Original Converted Value: {orig_converted:,.2f} {to_c}",
        f"- Current Converted Value: {new_converted:,.2f} {to_c}",
        f"- Foreign Exchange Variance: {fx_variance:+,.2f} {to_c} ({gain_loss})"
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: Retrieve Period Rate Change Data directly from Ledger (PostgreSQL)
# ---------------------------------------------------------------------------
class RetrieveRateChangeDataInput(BaseModel):
    metric: str = Field(
        default="SUM(glaa)",
        description="SQL aggregation expression, e.g. 'SUM(glaa)' or 'COUNT(*)'",
    )
    period_type: str = Field(
        default="fiscal_year",
        description="Type of period: 'fiscal_year' (glfy) or 'fiscal_period' (glpn)",
    )
    period_1: int = Field(description="First period identifier, e.g. 21 for FY2021")
    period_2: int = Field(description="Second period identifier, e.g. 26 for FY2026")
    additional_filter: Optional[str] = Field(
        default=None,
        description="Optional SQL filter clause, e.g. gldct = '##'",
    )


@tool("retrieve_ledger_rate_change", args_schema=RetrieveRateChangeDataInput)
def retrieve_ledger_rate_change(
    metric: str = "SUM(glaa)",
    period_type: str = "fiscal_year",
    period_1: int = 21,
    period_2: int = 22,
    additional_filter: Optional[str] = None,
) -> str:
    """
    Retrieves comparative data from the General Ledger across two periods from the Data Warehouse
    and calculates the rate of change automatically.
    """
    from chat.utils import init_database

    period_col = "glfy"
    if period_type == "fiscal_period":
        period_col = "glpn"

    where_clause = f"{period_col} IN ({period_1}, {period_2})"
    if additional_filter:
        where_clause += f" AND ({additional_filter})"

    sql = f"""
    SELECT 
        {period_col} AS period,
        {metric} AS metric_value
    FROM staging.proddta_f0911_account_ledger
    WHERE {where_clause}
    GROUP BY {period_col}
    ORDER BY {period_col} ASC;
    """

    try:
        db = init_database()
        res = db.run(sql)
        data = {}
        if isinstance(res, str):
            try:
                rows = ast.literal_eval(res)
                for r in rows:
                    data[int(r[0])] = float(r[1]) if r[1] is not None else 0.0
            except Exception:
                return f"Retrieved query results:\n{res}\nSQL executed: `{sql.strip()}`"

        v1 = data.get(period_1, 0.0)
        v2 = data.get(period_2, 0.0)

        calc_summary = calculate_rate_of_change.invoke(
            {"old_value": v1, "new_value": v2, "label": f"{metric} (Period {period_1} vs {period_2})"}
        )

        lines = [
            "Ledger Data Retrieval & Rate Change Analysis:",
            f"- Period {period_1}: {v1:,.2f}",
            f"- Period {period_2}: {v2:,.2f}",
            f"- Rate of Change: {calc_summary}",
            f"- Source Query: {sql.strip()}"
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error retrieving rate change data from ledger: {e}"


FINANCIAL_TOOLS = [
    financial_calculator,
    calculate_rate_of_change,
    calculate_cagr,
    calculate_currency_rate_change,
    retrieve_ledger_rate_change,
]

FINANCIAL_TOOL_NAMES = [t.name for t in FINANCIAL_TOOLS]


# ---------------------------------------------------------------------------
# Helper 1: Automatic Rate of Change Extraction from SQL Results
# ---------------------------------------------------------------------------
def compute_rate_changes_from_sql_result(sql_result: str, user_query: str) -> str:
    """
    Parses SQL result rows and user query to automatically compute verified rate of change,
    growth percentages, and currency variations using the financial calculator tools.
    """
    if not sql_result or sql_result.strip() in ("", "[]", "None"):
        return ""

    calculations = []

    # 1. Currency conversion check
    q_lower = user_query.lower()
    if any(term in q_lower for term in ["exchange rate", "currency", "usd", "eur", "gbp", "foreign", "fx", "ugx"]):
        amounts = re.findall(r"[\$€£]?\b\d+(?:,\d{3})*(?:\.\d+)?\b", user_query)
        if amounts:
            try:
                amt = float(amounts[0].replace(",", "").replace("$", "").replace("€", "").replace("£", ""))
                curr = "USD"
                for c in ["USD", "EUR", "GBP", "KES", "RWF"]:
                    if c.lower() in q_lower:
                        curr = c
                        break
                fx_calc = calculate_currency_rate_change.invoke({"amount": amt, "from_currency": curr, "to_currency": "UGX"})
                calculations.append(f"Currency & Exchange Rate Analysis:\n{fx_calc}")
            except Exception:
                pass

    # 2. Sequential period calculation from SQL result
    try:
        clean = re.sub(r"Decimal\('([0-9\.\-]+)'\)", r"\1", sql_result)
        clean = re.sub(r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)", r"'\1-\2-\3'", clean)
        clean = re.sub(r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+)[^\)]*\)", r"'\1-\2-\3'", clean)
        rows = ast.literal_eval(clean)
        if isinstance(rows, list) and len(rows) >= 2:
            if all(isinstance(r, (tuple, list)) for r in rows):
                first_row = rows[0]
                period_idx = 0
                val_idx = None
                for idx in range(len(first_row)-1, -1, -1):
                    val = first_row[idx]
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        val_idx = idx
                        break

                if val_idx is not None and val_idx != period_idx:
                    calculations.append("Verified Period-over-Period Rate of Change Calculations:")
                    prev_val = None
                    prev_p = None
                    for r in rows[:10]:
                        p = r[period_idx]
                        v = float(r[val_idx]) if r[val_idx] is not None else 0.0
                        if prev_val is not None and prev_val != 0:
                            diff = v - prev_val
                            pct = (diff / abs(prev_val)) * 100.0
                            direction = "Increase" if diff > 0 else ("Decrease" if diff < 0 else "Unchanged")
                            sign = "+" if diff > 0 else ""
                            calculations.append(
                                f"  * Period {prev_p} -> {p}: {sign}{pct:.2f}% {direction} (Difference: {diff:+,.2f}, from {prev_val:,.2f} to {v:,.2f})"
                            )
                        prev_val = v
                        prev_p = p

                    if len(rows) >= 3:
                        first_v = float(rows[0][val_idx])
                        last_v = float(rows[-1][val_idx])
                        n_periods = len(rows) - 1
                        if first_v > 0 and last_v > 0:
                            cagr_res = calculate_cagr.invoke({
                                "start_value": first_v,
                                "end_value": last_v,
                                "periods": float(n_periods),
                                "label": "Multi-period CAGR"
                            })
                            calculations.append(f"  * {cagr_res}")
    except Exception:
        pass

    if calculations:
        return "\n".join(calculations)
    return ""


# ---------------------------------------------------------------------------
# Helper 2: Direct Calculation & Math Engine
# ---------------------------------------------------------------------------
def _parse_numeric_str(s: str) -> float:
    """Parse numeric strings including commas, k/m/b/thousand/million/billion suffixes."""
    s = s.strip().lower().replace(",", "")
    mult = 1.0
    if s.endswith("billion") or s.endswith("b"):
        mult = 1e9
        s = s.replace("billion", "").replace("b", "")
    elif s.endswith("million") or s.endswith("m"):
        mult = 1e6
        s = s.replace("million", "").replace("m", "")
    elif s.endswith("thousand") or s.endswith("k"):
        mult = 1e3
        s = s.replace("thousand", "").replace("k", "")
    return float(s.strip()) * mult


def parse_and_compute_calculation_request(query: str) -> str:
    """
    Evaluates direct calculations, arithmetic expressions, percentage calculations,
    rate-of-change, CAGR, and currency conversions asked in natural language without database queries.
    """
    q = query.lower().strip()

    # 1. Rate of change between two explicit numbers: e.g. "from X to Y" or "between X and Y"
    match_from_to = re.findall(
        r"(?:from|between)\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+(?:to|and)\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)",
        q,
    )
    if match_from_to and any(w in q for w in ["rate", "change", "growth", "percentage", "variance", "difference", "increase", "decrease"]):
        try:
            v1 = _parse_numeric_str(match_from_to[0][0])
            v2 = _parse_numeric_str(match_from_to[0][1])
            return calculate_rate_of_change.invoke({"old_value": v1, "new_value": v2, "label": "Rate of Change"})
        except Exception:
            pass

    # 2. CAGR request: e.g. "cagr from 10m to 30m over 5 years"
    cagr_match = re.search(
        r"(?:cagr|compound\s+annual\s+growth)\s+(?:from\s+)?([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+to\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+(?:over|in|for)\s+(\d+(?:\.\d+)?)\s*(?:years?|periods?)",
        q,
    )
    if cagr_match:
        try:
            v1 = _parse_numeric_str(cagr_match.group(1))
            v2 = _parse_numeric_str(cagr_match.group(2))
            n = float(cagr_match.group(3))
            return calculate_cagr.invoke({"start_value": v1, "end_value": v2, "periods": n, "label": "CAGR"})
        except Exception:
            pass

    # 3. Currency conversion request: e.g. "convert 50,000 USD to UGX"
    curr_match = re.search(r"([\d,\.]+)\s*(usd|eur|gbp|kes|rwf|ugx)", q)
    if curr_match and any(w in q for w in ["convert", "exchange", "rate", "fx", "currency"]):
        try:
            amt = float(curr_match.group(1).replace(",", ""))
            curr = curr_match.group(2).upper()
            h_match = re.search(r"(?:historical|old|from rate)\s*(?:of|was|is|at)?\s*([\d,\.]+)", q)
            h_rate = float(h_match.group(1).replace(",", "")) if h_match else None
            return calculate_currency_rate_change.invoke({"amount": amt, "from_currency": curr, "to_currency": "UGX", "historical_rate": h_rate})
        except Exception:
            pass

    # 4. Percentage calculations: e.g. "what is 15% of 200,000,000", "15 percent of 500m"
    pct_match = re.search(
        r"(?:what\s+is\s+|calculate\s+|find\s+)?([\d,\.]+)\s*(?:%|percent)\s+(?:of\s+)([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)",
        q,
    )
    if pct_match:
        try:
            pct_val = float(pct_match.group(1).replace(",", ""))
            base_val = _parse_numeric_str(pct_match.group(2))
            res = (pct_val / 100.0) * base_val
            return (
                f"**Calculation Result:**\n\n"
                f"{pct_val}% of {base_val:,.2f} = **{res:,.2f}**\n\n"
                f"*(Formula: {pct_val} / 100 × {base_val:,.2f})*"
            )
        except Exception:
            pass

    # 5. Percentage Increase / Decrease: e.g. "increase 500,000 by 15%", "5000000 + 10%"
    inc_dec_match = re.search(
        r"(increase|decrease|discount|add|deduct)\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+by\s+([\d,\.]+)\s*(?:%|percent)",
        q,
    )
    if inc_dec_match:
        try:
            action = inc_dec_match.group(1)
            base_val = _parse_numeric_str(inc_dec_match.group(2))
            pct_val = float(inc_dec_match.group(3).replace(",", ""))
            delta = (pct_val / 100.0) * base_val
            if action in ("increase", "add"):
                res = base_val + delta
                desc = f"Increased {base_val:,.2f} by {pct_val}% (+{delta:,.2f})"
            else:
                res = base_val - delta
                desc = f"Decreased {base_val:,.2f} by {pct_val}% (-{delta:,.2f})"
            return (
                f"**Calculation Result:**\n\n"
                f"{desc} = **{res:,.2f}**\n\n"
                f"*(Delta: {delta:,.2f})*"
            )
        except Exception:
            pass

    # 6. Natural Language Word Math: "divide X by Y", "multiply X by Y", "add X and Y", "subtract X from Y"
    math_words = [
        (r"divide\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+by\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)", "/"),
        (r"multiply\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+(?:by|with)\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)", "*"),
        (r"add\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+(?:and|to)\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)", "+"),
        (r"subtract\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)\s+from\s+([\d,\.]+(?:\s*(?:billion|million|thousand|b|m|k))?)", "-_rev"),
    ]
    for pattern, op in math_words:
        m = re.search(pattern, q)
        if m:
            try:
                n1 = _parse_numeric_str(m.group(1))
                n2 = _parse_numeric_str(m.group(2))
                if op == "+":
                    res = n1 + n2
                    expr_str = f"{n1:,.2f} + {n2:,.2f}"
                elif op == "*":
                    res = n1 * n2
                    expr_str = f"{n1:,.2f} * {n2:,.2f}"
                elif op == "/":
                    if n2 == 0:
                        return "Cannot divide by zero."
                    res = n1 / n2
                    expr_str = f"{n1:,.2f} / {n2:,.2f}"
                elif op == "-_rev":
                    res = n2 - n1
                    expr_str = f"{n2:,.2f} - {n1:,.2f}"
                return f"**Calculation Result:**\n\n`{expr_str}` = **{res:,.4f}**".rstrip("0").rstrip(".") if isinstance(res, float) else f"**Calculation Result:**\n\n`{expr_str}` = **{res:,}**"
            except Exception:
                pass

    # 7. Direct math expression: e.g. "(250 - 180) / 180 * 100", "5000000 * 0.15", "125 * 360", "2^8"
    if re.search(r"[\d\)]\s*[\+\-\*/\^]\s*[\d\(]", query):
        expr_match = re.search(r"[\d\.\(\)\+\-\*/\s\^]+", query)
        if expr_match:
            expr_str = expr_match.group(0).strip()
            # Avoid single numbers or simple years (e.g. "2025")
            if any(op in expr_str for op in ["+", "-", "*", "/", "^"]):
                expr_str = expr_str.replace("^", "**")
                return financial_calculator.invoke({"expression": expr_str})

    return ""

