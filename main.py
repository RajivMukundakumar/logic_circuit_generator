import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy
from sympy.logic import SOPform, POSform
from sympy.parsing.sympy_parser import parse_expr
import schemdraw
from schemdraw.parsing import logicparse
import re
import math
import os
import json
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LogicRequest(BaseModel):
    query: str = ""
    image_b64: str = ""       # Base64 encoded image string
    image_mime: str = "image/png"
    gate_type: str = "standard"  # "standard", "nand", "nor"

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Logic Circuit & Gemini API is active."}

def preprocess_boolean_expr(expr_str: str) -> str:
    s = expr_str.strip()
    s = re.sub(r'\bAND\b', '&', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOR\b', '|', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNOT\b', '~', s, flags=re.IGNORECASE)
    s = s.replace('+', '|').replace('*', '&')
    
    while "'" in s:
        s = re.sub(r"([A-Za-z0-9_]+)'", r"~\1", s)
        s = re.sub(r"(\([^\(\)]+\))'", r"~\1", s)
        
    pattern = r'([A-Za-z0-9_]+|\))\s*([A-Za-z0-9_~\(])'
    for _ in range(3):
        def repl(m):
            if m.group(2) in ['|', '&']:
                return m.group(0)
            return f"{m.group(1)} & {m.group(2)}"
        s = re.sub(pattern, repl, s)
        
    return s

def try_deterministic_parse(query_str: str):
    query_str = query_str.strip()
    if not query_str:
        return None, None

    match_m = re.search(r'm\(([\d,\s]+)\)', query_str, re.IGNORECASE)
    if match_m:
        minterms = [int(x.strip()) for x in match_m.group(1).split(',')]
        max_val = max(minterms) if minterms else 0
        num_vars = max(len(bin(max_val)) - 2, 2)
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    match_M = re.search(r'M\(([\d,\s]+)\)', query_str, re.IGNORECASE)
    if match_M:
        maxterms = set(int(x.strip()) for x in match_M.group(1).split(','))
        max_val = max(maxterms) if maxterms else 0
        num_vars = max(len(bin(max_val)) - 2, 2)
        minterms = [i for i in range(2**num_vars) if i not in maxterms]
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    clean_query = query_str.replace(",", "").replace(" ", "")
    if all(c in '01' for c in clean_query) and len(clean_query) >= 2 and math.log2(len(clean_query)).is_integer():
        minterms = [i for i, bit in enumerate(clean_query) if bit == '1']
        num_vars = max(int(math.log2(len(clean_query))), 2)
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    try:
        sympy_str = preprocess_boolean_expr(query_str)
        expr = parse_expr(sympy_str)
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        if symbols:
            num_vars = len(symbols)
            minterms = []
            for i in range(2**num_vars):
                binary_vals = format(i, f'0{num_vars}b')
                sub_dict = {sym: bool(int(bit)) for sym, bit in zip(symbols, binary_vals)}
                if bool(expr.subs(sub_dict)):
                    minterms.append(i)
            return minterms, symbols
    except Exception:
        pass

    return None, None

def parse_with_gemini(query: str, image_b64: str, image_mime: str):
    """Uses server environment variable GEMINI_API_KEY to process requests."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is not configured on the Render server."
        )

    # Updated to gemini-3.6-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={key}"
    
    prompt = """
    You are a Digital Logic Design assistant. Analyze the given logic input (which may be a word problem, photo, screenshot, truth table, or expression).
    Perform the following tasks:
    1. Identify all input variables and map them to uppercase letters (e.g. A, B, C).
    2. Determine all minterm decimal values where the output Y = 1.
    3. Provide an ordered, step-by-step breakdown explaining how you parsed the input and derived the minterm states.

    Respond STRICTLY in JSON format with no markdown wrappers:
    {
        "variables": ["A", "B", "C"],
        "minterms": [1, 2, 5],
        "steps": [
            "Step 1: Problem Definition - ...",
            "Step 2: Variable Identification - ...",
            "Step 3: Truth Table & Minterm Extraction - ...",
            "Step 4: Expression Reduction & Gate Mapping - ..."
        ]
    }
    """

    parts = [{"text": prompt}]
    if query:
        parts.append({"text": f"User Input Query / Word Problem: {query}"})
    if image_b64:
        parts.append({
            "inline_data": {
                "mime_type": image_mime,
                "data": image_b64
            }
        })

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    try:
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Gemini API Error: {res.text}")
        
        result_json = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
        
        var_symbols = [sympy.Symbol(v) for v in result_json.get("variables", ["A", "B"])]
        minterms = result_json.get("minterms", [])
        steps = result_json.get("steps", [])
        
        return minterms, var_symbols, steps
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process input with Gemini: {str(e)}")

def generate_truth_table(var_symbols, minterms):
    num_vars = len(var_symbols)
    headers = [str(v) for v in var_symbols] + ["Y"]
    rows = []
    minterm_set = set(minterms)
    
    for i in range(2**num_vars):
        binary_str = format(i, f'0{num_vars}b')
        row_bits = [int(b) for b in binary_str]
        rows.append(row_bits + [1 if i in minterm_set else 0])
        
    return {"headers": headers, "rows": rows}

def generate_kmap_data(var_symbols, minterms):
    n = len(var_symbols)
    minterm_set = set(minterms)
    gray_2 = [0, 1]
    gray_4 = [0, 1, 3, 2]
    
    if n == 2:
        row_vars, col_vars = str(var_symbols[0]), str(var_symbols[1])
        row_labels, col_labels = ["0", "1"], ["0", "1"]
        grid = [[1 if (r*2 + c) in minterm_set else 0 for c in gray_2] for r in gray_2]
    elif n == 3:
        row_vars = str(var_symbols[0])
        col_vars = f"{var_symbols[1]}{var_symbols[2]}"
        row_labels = ["0", "1"]
        col_labels = ["00", "01", "11", "10"]
        grid = [[1 if (r*4 + c) in minterm_set else 0 for c in gray_4] for r in gray_2]
    else:
        row_vars = f"{var_symbols[0]}{var_symbols[1]}"
        col_vars = f"{var_symbols[2]}{var_symbols[3]}" if n >= 4 else "CD"
        row_labels = ["00", "01", "11", "10"]
        col_labels = ["00", "01", "11", "10"]
        grid = [[1 if (r*4 + c) in minterm_set else 0 for c in gray_4] for r in gray_4]

    return {
        "num_vars": n,
        "row_vars": row_vars,
        "col_vars": col_vars,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "grid": grid
    }

def sym_to_nand_str(expr):
    if isinstance(expr, sympy.Symbol):
        return str(expr)
    if isinstance(expr, sympy.Not):
        return f"not ({expr.args[0]} and {expr.args[0]})"

    def nand_literal(lit):
        if isinstance(lit, sympy.Symbol):
            return str(lit)
        elif isinstance(lit, sympy.Not):
            return f"not ({lit.args[0]} and {lit.args[0]})"
        return str(lit)

    def nand_term(term):
        if isinstance(term, sympy.And):
            literals = [nand_literal(arg) for arg in term.args]
            return f"not ({' and '.join(literals)})"
        return nand_literal(term)

    if isinstance(expr, sympy.Or):
        terms = [nand_term(arg) for arg in expr.args]
        return f"not ({' and '.join(terms)})"
    elif isinstance(expr, sympy.And):
        term = nand_term(expr)
        return f"not ({term} and {term})"
    
    return str(expr)

def sym_to_nor_str(expr):
    if isinstance(expr, sympy.Symbol):
        return str(expr)
    if isinstance(expr, sympy.Not):
        return f"not ({expr.args[0]} or {expr.args[0]})"

    def nor_literal(lit):
        if isinstance(lit, sympy.Symbol):
            return str(lit)
        elif isinstance(lit, sympy.Not):
            return f"not ({lit.args[0]} or {lit.args[0]})"
        return str(lit)

    def nor_term(term):
        if isinstance(term, sympy.Or):
            literals = [nor_literal(arg) for arg in term.args]
            return f"not ({' or '.join(literals)})"
        return nor_literal(term)

    if isinstance(expr, sympy.And):
        terms = [nor_term(arg) for arg in expr.args]
        return f"not ({' or '.join(terms)})"
    elif isinstance(expr, sympy.Or):
        term = nor_term(expr)
        return f"not ({term} or {term})"
    
    return str(expr)

@app.post("/solve")
def solve_logic(request: LogicRequest):
    try:
        query = request.query.strip()
        image_b64 = request.image_b64.strip()
        gate_type = request.gate_type.lower()
        steps = []

        if not query and not image_b64:
            raise HTTPException(status_code=400, detail="Please enter text/word problem or upload an image.")

        minterms = None
        var_symbols = None

        if not image_b64:
            minterms, var_symbols = try_deterministic_parse(query)

        if minterms is None or var_symbols is None:
            minterms, var_symbols, steps = parse_with_gemini(query, image_b64, request.image_mime)
        else:
            steps = [
                "Step 1: Direct Logic Extraction - Recognized expression, minterms, or truth table format.",
                f"Step 2: Variable Identification - Extracted variables {[str(v) for v in var_symbols]}.",
                f"Step 3: Minterm Analysis - Active minterm states identified at indices {minterms}.",
                "Step 4: Truth Table & K-Map Generation - Calculated output states across all binary combinations."
            ]

        num_vars = len(var_symbols)
        truth_table = generate_truth_table(var_symbols, minterms)
        kmap_data = generate_kmap_data(var_symbols, minterms)

        if not minterms:
            sop_str, pos_str = "0", "0"
        elif len(minterms) == (2 ** num_vars):
            sop_str, pos_str = "1", "1"
        else:
            sop_expr = SOPform(var_symbols, minterms)
            pos_expr = POSform(var_symbols, minterms)
            sop_str, pos_str = str(sop_expr), str(pos_expr)

        if sop_str in ["0", "1"]:
            svg_string = f"<svg width='240' height='60'><text x='20' y='35' font-family='sans-serif' font-size='16'>Output = {sop_str}</text></svg>"
        else:
            sop_expr = SOPform(var_symbols, minterms)
            if gate_type == "nand":
                schem_expression = sym_to_nand_str(sop_expr)
            elif gate_type == "nor":
                schem_expression = sym_to_nor_str(POSform(var_symbols, minterms))
            else:
                schem_expression = str(sop_expr).replace('&', 'and').replace('|', 'or').replace('~', 'not ')

            try:
                drawing = logicparse(schem_expression)
                svg_bytes = drawing.get_imagedata('svg')
                svg_string = svg_bytes.decode('utf-8')
            except Exception as e:
                svg_string = f"<p class='text-red-500 font-mono text-sm'>Schematic generation error: {str(e)}</p>"

        return {
            "sop": sop_str,
            "pos": pos_str,
            "steps": steps,
            "truth_table": truth_table,
            "kmap": kmap_data,
            "svg": svg_string
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
