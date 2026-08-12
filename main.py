from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy
from sympy.logic import SOPform, POSform
import schemdraw
from schemdraw.parsing import logicparse
import re
import math

app = FastAPI()

# Configure CORS cleanly for production
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
    """Root route so Render health pings return HTTP 200 OK instead of 404."""
    return {"status": "ok", "message": "Logic Circuit Generator API is active."}

def parse_input_to_minterms(query: str):
    query = query.strip()

    # Format 1: Minterm notation e.g., m(1,2,5)
    match_m = re.search(r'm\(([\d,\s]+)\)', query, re.IGNORECASE)
    if match_m:
        minterms = [int(x.strip()) for x in match_m.group(1).split(',')]
        max_val = max(minterms) if minterms else 0
        num_vars = max(len(bin(max_val)) - 2, 2)
        return minterms, num_vars

    # Format 2: Truth Table output array e.g., "0,1,1,1"
    clean_query = query.replace(",", "").replace(" ", "")
    if all(c in '01' for c in clean_query) and len(clean_query) > 1:
        minterms = [i for i, bit in enumerate(clean_query) if bit == '1']
        rows = len(clean_query)
        num_vars = max(math.ceil(math.log2(rows)), 2)
        return minterms, num_vars

    return None, None

def generate_variable_symbols(num_vars: int):
    if num_vars <= 26:
        return [sympy.Symbol(chr(65 + i)) for i in range(num_vars)]
    return [sympy.Symbol(f"X{i}") for i in range(num_vars)]

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
async def solve_logic(request: LogicRequest):
    try:
        query = request.query
        gate_type = request.gate_type.lower()
        
        minterms, num_vars = parse_input_to_minterms(query)
        
        if minterms is None:
            raise HTTPException(
                status_code=400, 
                detail="Invalid input format. Enter truth table outputs (e.g. 0,1,1,0) or minterms m(0,1,3)."
            )

        var_symbols = generate_variable_symbols(num_vars)
        
        # Ground case
        if not minterms:
            return {
                "sop": "0", 
                "pos": "0", 
                "explanation": "Output is constant 0 (Ground).",
                "svg": "<svg width='240' height='60'><text x='20' y='35' font-family='sans-serif' font-size='16'>Output = 0 (Ground)</text></svg>"
            }

        # VCC case
        if len(minterms) == (2 ** num_vars):
            return {
                "sop": "1", 
                "pos": "1", 
                "explanation": "Output is constant 1 (VCC).",
                "svg": "<svg width='240' height='60'><text x='20' y='35' font-family='sans-serif' font-size='16'>Output = 1 (VCC)</text></svg>"
            }

        sop_expr = SOPform(var_symbols, minterms)
        pos_expr = POSform(var_symbols, minterms)

        # Direct wire pass-through
        if isinstance(sop_expr, sympy.Symbol):
            var_name = str(sop_expr)
            return {
                "sop": var_name,
                "pos": var_name,
                "explanation": f"Simplified directly to input variable {var_name}. No logic gates required.",
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
        explanation = f"Processed {len(minterms)} minterms across {num_vars} variables ({var_names_str}). Implemented using {gate_type.upper()} gate logic."

        return {
            "sop": str(sop_expr),
            "pos": str(pos_expr),
            "explanation": explanation,
            "svg": svg_string
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server calculation error: {str(e)}")
