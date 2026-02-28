"""
tools/calculator.py
"""

from langchain.tools import tool


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression. Input should be a valid Python
    math expression as a string, e.g. "2 ** 10" or "sum([1, 2, 3])".
    """
    try:
        # Safe eval: only allow math builtins
        allowed = {k: v for k, v in __builtins__.items() if k in (
            "abs", "round", "min", "max", "sum", "pow", "divmod"
        )} if isinstance(__builtins__, dict) else {}
        import math
        allowed.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
