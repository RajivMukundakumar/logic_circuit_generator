import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy
from sympy.logic import SOPform, POSform
from sympy.parsing.sympy_parser import parse_expr
import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm
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

def get_pin(g, pin_name):
    """Safely retrieves an anchor/pin from a schemdraw element across various versions."""
    if hasattr(g, pin_name):
        return getattr(g, pin_name)
    if hasattr(g, 'anchors') and pin_name in g.anchors:
        return g.anchors[pin_name]
    if hasattr(g, 'anchors'):
        for k, v in g.anchors.items():
            if k.lower() == pin_name.lower():
                return v
    raise AttributeError(f"Anchor '{pin_name}' not defined in Element. Available: {list(getattr(g, 'anchors', {}).keys())}")

def extract_terms(expr, outer_cls, inner_cls):
    if expr in [sympy.S.Zero, 0, sympy.S.One, 1]:
        return None

    terms = expr.args if isinstance(expr, outer_cls) else [expr]
    result = []
    for term in terms:
        literals = []
        if isinstance(term, inner_cls):
            for arg in term.args:
                if isinstance(arg, sympy.Not):
                    literals.append((arg.args[0], True))
                else:
                    literals.append((arg, False))
        elif isinstance(term, sympy.Not):
            literals.append((term.args[0], True))
        else:
            literals.append((term, False))
        result.append(literals)
    return result

def build_term_gate_2input(d, lits, GateCls, inverter_out, x_start, y_start):
    if len(lits) == 1:
        var, neg = lits[0]
        if neg:
            return inverter_out[str(var)], x_start
        else:
            p = (x_start, y_start)
            d += elm.Line().at((x_start - 1.2, y_start)).to(p).label(str(var), loc='left')
            return p, x_start

    if len(lits) == 2:
        g = GateCls(inputs=2).at((x_start, y_start))
        d += g
        for i, (var, neg) in enumerate(lits):
            in_anchor = get_pin(g, f"in{i + 1}")
            if neg:
                d += elm.Line().at(inverter_out[str(var)]).to(in_anchor)
            else:
                d += elm.Line().at((x_start - 1.2, in_anchor[1])).to(in_anchor).label(str(var), loc='left')
        return g.out, x_start

    lit_anchors = []
    lit_y_step = 0.8
    base_y = y_start + (len(lits) - 1) * lit_y_step / 2.0

    for idx, (var, neg) in enumerate(lits):
        ly = base_y - idx * lit_y_step
        if neg:
            lit_anchors.append((inverter_out[str(var)], ly))
        else:
            p_in = (x_start - 1.2, ly)
            p_out = (x_start, ly)
            d += elm.Line().at(p_in).to(p_out).label(str(var), loc='left')
            lit_anchors.append((p_out, ly))

    curr = lit_anchors
    cx = x_start + 1.2
    while len(curr) > 1:
        nxt = []
        i = 0
        while i < len(curr):
            if i + 1 < len(curr):
                anc1, y1 = curr[i]
                anc2, y2 = curr[i+1]
                my = (y1 + y2) / 2.0
                g = GateCls(inputs=2).at((cx, my))
                d += g
                d += elm.Line().at(anc1).to(get_pin(g, "in1"))
                d += elm.Line().at(anc2).to(get_pin(g, "in2"))
                nxt.append((g.out, my))
                i += 2
            else:
                anc, y = curr[i]
                nxt.append((anc, y))
                i += 1
        curr = nxt
        cx += 2.5
    return curr[0][0], cx - 2.5

def reduce_term_outputs_2input(d, term_outputs, FinalCls, x_start, x_step=3.0):
    current = term_outputs[:]
    curr_x = x_start

    while len(current) > 1:
        next_level = []
        i = 0
        while i < len(current):
            if i + 1 < len(current):
                anc1, y1 = current[i]
                anc2, y2 = current[i+1]
                mid_y = (y1 + y2) / 2.0

                g = FinalCls(inputs=2).at((curr_x, mid_y))
                d += g

                d += elm.Line().at(anc1).to(get_pin(g, "in1"))
                d += elm.Line().at(anc2).to(get_pin(g, "in2"))

                next_level.append((g.out, mid_y))
                i += 2
            else:
                anc, y = current[i]
                next_level.append((anc, y))
                i += 1

        current = next_level
        curr_x += x_step

    return current[0][0], curr_x - x_step

def build_two_level_svg(terms, gate_type):
    d = schemdraw.Drawing()
    row_h = 2.5

    if gate_type == "nand":
        GateCls = logic.Nand
        FinalCls = logic.Nand
        invert_literal_via_gate = True
    elif gate_type == "nor":
        GateCls = logic.Nor
        FinalCls = logic.Nor
        invert_literal_via_gate = True
    else:
        GateCls = logic.And
        FinalCls = logic.Or
        invert_literal_via_gate = False

    inv_vars, seen = [], set()
    for lits in terms:
        for var, neg in lits:
            if neg and str(var) not in seen:
                seen.add(str(var))
                inv_vars.append(str(var))

    inverter_out = {}
    y = 0
    for var in inv_vars:
        if invert_literal_via_gate:
            g = GateCls(inputs=2).at((0, y)).label(f"{var}'", loc='right')
            d += elm.Line().at((-1.5, y + 0.3)).to(get_pin(g, "in1")).label(var, loc='left')
            d += elm.Line().at((-1.5, y - 0.3)).to(get_pin(g, "in2"))
            d += g
        else:
            g = logic.Not().at((0, y)).label(f"{var}'", loc='right')
            d += elm.Line().at((-1.5, y)).to(get_pin(g, "in1")).label(var, loc='left')
            d += g
        inverter_out[var] = g.out
        y -= row_h

    x1 = 3.5
    term_outputs = []
    y = 0
    max_x_term = x1

    for lits in terms:
        out_anchor, last_x = build_term_gate_2input(d, lits, GateCls, inverter_out, x1, y)
        term_outputs.append((out_anchor, y))
        max_x_term = max(max_x_term, last_x)
        y -= row_h

    x2 = max_x_term + 3.0

    if len(term_outputs) == 1:
        if invert_literal_via_gate:
            mid_y = term_outputs[0][1]
            final = FinalCls(inputs=2).at((x2, mid_y))
            d += elm.Line().at(term_outputs[0][0]).to(get_pin(final, "in1"))
            d += elm.Line().at(term_outputs[0][0]).to(get_pin(final, "in2"))
            d += final
            d += elm.Line().at(final.out).right().label('Y', loc='right')
        else:
            d += elm.Line().at(term_outputs[0][0]).right().label('Y', loc='right')
        return d.get_imagedata('svg').decode('utf-8')

    final_out, _ = reduce_term_outputs_2input(d, term_outputs, FinalCls, x2, x_step=3.0)
    d += elm.Line().at(final_out).right().label('Y', loc='right')

    return d.get_imagedata('svg').decode('utf-8')

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
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={key}"

            prompt = """
            You are a Digital Logic Design assistant. Analyze the given logic input (word problem, photo, screenshot, truth table, or K-map).
            Perform the following tasks:
            1. Identify all input variables and map them in order (e.g. A, B, C, D).
            2. Determine all minterm decimal values where the output Y = 1.
            3. Provide an ordered, step-by-step breakdown explaining how you derived the minterm states.

            Respond STRICTLY in JSON format with no markdown wrappers:
            {
                "variables": ["A", "B", "C", "D"],
                "minterms": [2, 3, 5, 6, 7, 13, 15],
                "steps": [
                    "Step 1: Analyzed 4-variable K-map grid with AB rows and CD columns.",
                    "Step 2: Located all '1' cells across Gray-code indices.",
                    "Step 3: Extracted minterms: m(2, 3, 5, 6, 7, 13, 15)."
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
            try:
                if gate_type == "nand":
                    terms = extract_terms(SOPform(var_symbols, minterms), sympy.Or, sympy.And)
                elif gate_type == "nor":
                    terms = extract_terms(POSform(var_symbols, minterms), sympy.And, sympy.Or)
                else:
                    terms = extract_terms(SOPform(var_symbols, minterms), sympy.Or, sympy.And)

                svg_string = build_two_level_svg(terms, gate_type)
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
