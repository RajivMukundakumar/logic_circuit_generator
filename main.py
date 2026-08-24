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
    image_b64: str = ""
    image_mime: str = "image/png"
    gate_type: str = "standard"

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Logic Circuit API active."}

# --- Strict 2-Input NAND Tree Builder ---
def build_2input_nand_tree(items):
    if not items:
        return "1"
    if len(items) == 1:
        return f"not ({items[0]} and {items[0]})"
    if len(items) == 2:
        return f"not ({items[0]} and {items[1]})"
    mid = len(items) // 2
    left_nand = build_2input_nand_tree(items[:mid])
    right_nand = build_2input_nand_tree(items[mid:])
    left_and = f"not ({left_nand} and {left_nand})"
    right_and = f"not ({right_nand} and {right_nand})"
    return f"not ({left_and} and {right_and})"

def get_nand_literal(lit):
    if isinstance(lit, sympy.Symbol):
        return str(lit)
    elif isinstance(lit, sympy.Not):
        return f"not ({lit.args[0]} and {lit.args[0]})"
    return str(lit)

def sym_to_nand_str(expr):
    if expr in [sympy.S.Zero, 0]:
        return "0"
    if expr in [sympy.S.One, 1]:
        return "1"
    
    terms = expr.args if isinstance(expr, sympy.Or) else [expr]
    not_P_terms = []

    for term in terms:
        if isinstance(term, sympy.And):
            lits = [get_nand_literal(arg) for arg in term.args]
        else:
            lits = [get_nand_literal(term)]
        not_P_terms.append(build_2input_nand_tree(lits))

    return build_2input_nand_tree(not_P_terms)

# --- Strict 2-Input NOR Tree Builder ---
def build_2input_nor_tree(items):
    if not items:
        return "0"
    if len(items) == 1:
        return f"not ({items[0]} or {items[0]})"
    if len(items) == 2:
        return f"not ({items[0]} or {items[1]})"
    mid = len(items) // 2
    left_nor = build_2input_nor_tree(items[:mid])
    right_nor = build_2input_nor_tree(items[mid:])
    left_or = f"not ({left_nor} or {left_nor})"
    right_or = f"not ({right_nor} or {right_nor})"
    return f"not ({left_or} or {right_or})"

def get_nor_literal(lit):
    if isinstance(lit, sympy.Symbol):
        return str(lit)
    elif isinstance(lit, sympy.Not):
        return f"not ({lit.args[0]} or {lit.args[0]})"
    return str(lit)

def sym_to_nor_str(expr):
    if expr in [sympy.S.Zero, 0]:
        return "0"
    if expr in [sympy.S.One, 1]:
        return "1"

    sum_terms = expr.args if isinstance(expr, sympy.And) else [expr]
    not_S_terms = []

    for term in sum_terms:
        if isinstance(term, sympy.Or):
            lits = [get_nor_literal(arg) for arg in term.args]
        else:
            lits = [get_nor_literal(term)]
        not_S_terms.append(build_2input_nor_tree(lits))

    return build_2input_nor_tree(not_S_terms)

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

def parse_with_gemini_safe(query: str, image_b64: str, image_mime: str):
    key = os.environ.get("GEMINI_API_KEY", "")
    gemini_error = ""

    if image_b64:
        if "," in image_b64:
            header, image_b64 = image_b64.split(",", 1)
            if "data:" in header and ";" in header:
                image_mime = header.split(";")[0].replace("data:", "")
        image_b64 = re.sub(r'\s+', '', image_b64)

    if not image_mime or not image_mime.startswith("image/"):
        image_mime = "image/png"

    if key:
        try:
            # Using Gemini 3.6 Flash endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={key}"
            
            prompt = """
            You are a Digital Logic Design assistant. Analyze the given logic input (word problem, photo, screenshot, truth table, or expression).
            Perform the following tasks:
            1. Identify all input variables and map them to uppercase letters (e.g. A, B, C, D).
            2. Determine all minterm decimal values where the output Y = 1.
            3. Provide an ordered, step-by-step breakdown explaining how you derived the minterm states.

            Respond STRICTLY in JSON format with no markdown wrappers:
            {
                "variables": ["A", "B", "C", "D"],
                "minterms": [1, 2, 5],
                "steps": [
                    "Step 1: Identified input variables from the diagram or truth table.",
                    "Step 2: Located all truth table rows where output Y = 1.",
                    "Step 3: Extracted minterms as decimal indices."
                ]
            }
            """

            parts = [{"text": prompt}]
            if query.strip():
                parts.append({"text": f"User Input Query: {query}"})
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

            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
            if res.status_code == 200:
                result_json = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
                var_symbols = [sympy.Symbol(v) for v in result_json.get("variables", ["A", "B", "C", "D"])]
                minterms = result_json.get("minterms", [])
                steps = result_json.get("steps", [])
                return minterms, var_symbols, steps
            else:
                gemini_error = f"Gemini API returned status code {res.status_code}: {res.text}"
        except Exception as e:
            gemini_error = str(e)
    else:
        gemini_error = "GEMINI_API_KEY environment variable is missing or empty."

    raw_vars = sorted(list(set(re.findall(r'[A-Za-z]', query.upper()))))
    valid_vars = [v for v in raw_vars if v in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"][:4]
    if not valid_vars:
        valid_vars = ["A", "B", "C", "D"]

    var_symbols = [sympy.Symbol(v) for v in valid_vars]
    minterms = []  

    steps = [
        f"⚠️ Vision Processing Error: {gemini_error}",
        "Fallback Execution: Unable to parse freeform text/image without API response.",
        f"Assumed default variables: {[str(v) for v in var_symbols]}."
    ]

    return minterms, var_symbols, steps

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

@app.post("/solve")
def solve_logic(request: LogicRequest):
    try:
        query = request.query.strip()
        image_b64 = request.image_b64.strip()
        gate_type = request.gate_type.lower()
        steps = []

        if not query and not image_b64:
            raise HTTPException(status_code=400, detail="Please enter text or upload an image.")

        minterms = None
        var_symbols = None

        if not image_b64:
            minterms, var_symbols = try_deterministic_parse(query)

        if minterms is None or var_symbols is None:
            minterms, var_symbols, steps = parse_with_gemini_safe(query, image_b64, request.image_mime)
        else:
            steps = [
                "Step 1: Direct Parsing - Identified input minterm sequence.",
                f"Step 2: Variable Assignment - Mapped inputs to {[str(v) for v in var_symbols]}.",
                f"Step 3: Minterm States - Active outputs at indices {minterms}.",
                "Step 4: Truth Table & K-Map Generation - Calculated full output matrix."
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
            if gate_type == "nand":
                schem_expression = sym_to_nand_str(SOPform(var_symbols, minterms))
            elif gate_type == "nor":
                schem_expression = sym_to_nor_str(POSform(var_symbols, minterms))
            else:
                schem_expression = str(SOPform(var_symbols, minterms)).replace('&', 'and').replace('|', 'or').replace('~', 'not ')

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
