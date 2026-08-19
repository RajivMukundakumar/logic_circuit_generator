import matplotlib
# Force Matplotlib to non-interactive mode for server stability
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LogicRequest(BaseModel):
    query: str
    gate_type: str = "standard"  # "standard", "nand", "nor"

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Logic Circuit Generator API is active."}

def preprocess_boolean_expr(expr_str: str) -> str:
    """Normalizes Boolean expressions (A'B + C, A AND B, (A+B)') into SymPy format."""
    s = expr_str.strip()
    
    # Replace text operators
    s = re.sub(r'\bAND\b', '&', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOR\b', '|', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNOT\b', '~', s, flags=re.IGNORECASE)
    
    # Replace standard notation symbols
    s = s.replace('+', '|').replace('*', '&')
    
    # Convert prime/apostrophe notation: A' -> ~A, (A|B)' -> ~(A|B)
    while "'" in s:
        s = re.sub(r"([A-Za-z0-9_]+)'", r"~\1", s)
        s = re.sub(r"(\([^\(\)]+\))'", r"~\1", s)
        
    # Handle implicit multiplication (e.g., AB -> A & B, ~AB -> ~A & B, (A|B)C -> (A|B) & C)
    pattern = r'([A-Za-z0-9_]+|\))\s*([A-Za-z0-9_~\(])'
    for _ in range(3):
        def repl(m):
            g1, g2 = m.group(1), m.group(2)
            if g2 in ['|', '&']:
                return m.group(0)
            return f"{g1} & {g2}"
        s = re.sub(pattern, repl, s)
        
    return s

def parse_input_to_minterms(query: str):
    query_str = query.strip()

    # Format 1: Minterm notation e.g. m(0,1,2,5)
    match_m = re.search(r'm\(([\d,\s]+)\)', query_str, re.IGNORECASE)
    if match_m:
        minterms = [int(x.strip()) for x in match_m.group(1).split(',')]
        max_val = max(minterms) if minterms else 0
        num_vars = max(len(bin(max_val)) - 2, 2)
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    # Format 2: Maxterm notation e.g. M(0,1,2)
    match_M = re.search(r'M\(([\d,\s]+)\)', query_str, re.IGNORECASE)
    if match_M:
        maxterms = set(int(x.strip()) for x in match_M.group(1).split(','))
        max_val = max(maxterms) if maxterms else 0
        num_vars = max(len(bin(max_val)) - 2, 2)
        minterms = [i for i in range(2**num_vars) if i not in maxterms]
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    # Format 3: Truth Table output binary string e.g. "0,1,1,0" or "0110"
    clean_query = query_str.replace(",", "").replace(" ", "")
    if all(c in '01' for c in clean_query) and len(clean_query) >= 2:
        minterms = [i for i, bit in enumerate(clean_query) if bit == '1']
        rows = len(clean_query)
        num_vars = max(math.ceil(math.log2(rows)), 2)
        var_symbols = [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
        return minterms, var_symbols

    # Format 4: Raw Boolean Expression e.g. A'B + C, A AND B, (A + B)'
    try:
        sympy_str = preprocess_boolean_expr(query_str)
        expr = parse_expr(sympy_str)
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        if not symbols:
            return None, None
        
        num_vars = len(symbols)
        minterms = []
        for i in range(2**num_vars):
            binary_vals = format(i, f'0{num_vars}b')
            sub_dict = {sym: bool(int(bit)) for sym, bit in zip(symbols, binary_vals)}
            res = bool(expr.subs(sub_dict))
            if res:
                minterms.append(i)
        return minterms, symbols
    except Exception:
        pass

    return None, None

def generate_truth_table(var_symbols, minterms):
    num_vars = len(var_symbols)
    headers = [str(v) for v in var_symbols] + ["Y"]
    rows = []
    minterm_set = set(minterms)
    
    for i in range(2**num_vars):
        binary_str = format(i, f'0{num_vars}b')
        row_bits = [int(b) for b in binary_str]
        output_bit = 1 if i in minterm_set else 0
        rows.append(row_bits + [output_bit])
        
    return {"headers": headers, "rows": rows}

def sym_to_nand_str(expr):
    if isinstance(expr, sympy.Symbol):
        return str(expr)
    if isinstance(expr, sympy.Not):
        var = str(expr.args[0])
        return f"not ({var} and {var})"

    def nand_literal(lit):
        if isinstance(lit, sympy.Symbol):
            return str(lit)
        elif isinstance(lit, sympy.Not):
            var = str(lit.args[0])
            return f"not ({var} and {var})"
        return str(lit)

    def nand_term(term):
        if isinstance(term, sympy.And):
            literals = [nand_literal(arg) for arg in term.args]
            return f"not ({' and '.join(literals)})"
        else:
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
        var = str(expr.args[0])
        return f"not ({var} or {var})"

    def nor_literal(lit):
        if isinstance(lit, sympy.Symbol):
            return str(lit)
        elif isinstance(lit, sympy.Not):
            var = str(lit.args[0])
            return f"not ({var} or {var})"
        return str(lit)

    def nor_term(term):
        if isinstance(term, sympy.Or):
            literals = [nor_literal(arg) for arg in term.args]
            return f"not ({' or '.join(literals)})"
        else:
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
        query = request.query
        gate_type = request.gate_type.lower()
        
        minterms, var_symbols = parse_input_to_minterms(query)
        
        if minterms is None or var_symbols is None:
            raise HTTPException(
                status_code=400, 
                detail="Invalid input format. Enter a Boolean expression (e.g. A'B + C), minterms m(0,1,2), maxterms M(0,1), or truth table outputs (0,1,1,0)."
            )

        num_vars = len(var_symbols)
        truth_table = generate_truth_table(var_symbols, minterms)

        # Ground case
        if not minterms:
            return {
                "sop": "0", 
                "pos": "0", 
                "explanation": "Output is constant 0 (Ground).",
                "truth_table": truth_table,
                "svg": "<svg width='240' height='60'><text x='20' y='35' font-family='sans-serif' font-size='16'>Output = 0 (Ground)</text></svg>"
            }

        # VCC case
        if len(minterms) == (2 ** num_vars):
            return {
                "sop": "1", 
                "pos": "1", 
                "explanation": "Output is constant 1 (VCC).",
                "truth_table": truth_table,
                "svg": "<svg width='240' height='60'><text x='20' y='35' font-family='sans-serif' font-size='16'>Output = 1 (VCC)</text></svg>"
            }

        sop_expr = SOPform(var_symbols, minterms)
        pos_expr = POSform(var_symbols, minterms)

        if isinstance(sop_expr, sympy.Symbol):
            var_name = str(sop_expr)
            return {
                "sop": var_name,
                "pos": var_name,
                "explanation": f"Simplified directly to input variable {var_name}. No logic gates required.",
                "truth_table": truth_table,
                "svg": f"<svg width='240' height='60'><line x1='20' y1='30' x2='180' y2='30' stroke='black' stroke-width='2'/><text x='5' y='35' font-family='sans-serif'>{var_name}</text><text x='190' y='35' font-family='sans-serif'>Y</text></svg>"
            }

        if gate_type == "nand":
            schem_expression = sym_to_nand_str(sop_expr)
        elif gate_type == "nor":
            schem_expression = sym_to_nor_str(pos_expr)
        else:
            schem_expression = str(sop_expr).replace('&', 'and').replace('|', 'or').replace('~', 'not ')

        try:
            drawing = logicparse(schem_expression)
            svg_bytes = drawing.get_imagedata('svg')
            svg_string = svg_bytes.decode('utf-8')
        except Exception as e:
            svg_string = f"<p class='text-red-500 font-mono text-sm'>Schematic generation error: {str(e)}</p>"

        var_names_str = ", ".join([str(v) for v in var_symbols])
        explanation = f"Processed {len(minterms)} active minterm states across {num_vars} variables ({var_names_str}). Output logic rendered using {gate_type.upper()} gate implementation."

        return {
            "sop": str(sop_expr),
            "pos": str(pos_expr),
            "explanation": explanation,
            "truth_table": truth_table,
            "svg": svg_string
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server calculation error: {str(e)}")
