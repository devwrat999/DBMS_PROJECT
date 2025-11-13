import re
from typing import List, Optional, Union
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Flask Server Setup ---
app = Flask(__name__)
# CORS is required to allow the HTML file to talk to this server
CORS(app) 

# ▼▼▼ YOUR ENTIRE BACKEND CODE (UNCHANGED) ▼▼▼
# ---------- Phase 1: The AST (Query Object Model) ----------
class Node:
    """Base class for all AST nodes."""
    def __repr__(self):
        return f"{self.__class__.__name__}(...)"

class TableScan(Node):
    """Represents a leaf node (a base table)."""
    def __init__(self, name: str):
        self.name = name
        self.alias = None

class Select(Node):
    """Represents a sigma (σ) operation."""
    def __init__(self, child: Node, predicate: str):
        self.child = child
        self.predicate = predicate

class Project(Node):
    """Represents a pi (π) operation."""
    def __init__(self, child: Node, columns: List[str]):
        self.child = child
        self.columns = columns

class Join(Node):
    """Represents a join (⋈) operation."""
    def __init__(self, left: Node, right: Node, condition: Optional[str] = None, join_type: str = 'INNER'):
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type

class Rename(Node):
    """Represents a rho (ρ) operation (table rename)."""
    def __init__(self, child: Node, new_name: str):
        self.child = child
        self.new_name = new_name

class Aggregate(Node):
    """Represents a gamma (γ) operation (GROUP BY)."""
    def __init__(self, child: Node, group_by_cols: List[str], aggregates: List[str]):
        self.child = child
        self.group_by_cols = group_by_cols
        self.aggregates = aggregates

class SetOperation(Node):
    """Base class for UNION, INTERSECT, EXCEPT."""
    def __init__(self, left: Node, right: Node, op_type: str, is_all: bool = False):
        self.left = left
        self.right = right
        self.op_type = op_type
        self.is_all = is_all

# ---------- Phase 2: The Parser (String -> AST) ----------
def _find_top_level_comma(s: str) -> int:
    depth = 0
    for i, c in enumerate(s):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ',' and depth == 0: return i
    return -1

def _parse_to_ast(expr: str) -> Node:
    expr = expr.strip()
    if re.search(r'[σπ∪∩×⋈ρ]', expr):
        raise ValueError("❌ Symbolic operators are not allowed. Please use RA keywords (e.g., 'sigma', 'pi', etc.).")

    def strip_outer_parens(s):
        s = s.strip()
        if s.startswith('(') and s.endswith(')'):
            count = 0
            for i, c in enumerate(s):
                if c == '(': count += 1
                elif c == ')': count -= 1
                if count == 0 and i < len(s) - 1: return s
            return strip_outer_parens(s[1:-1])
        return s

    expr = strip_outer_parens(expr)
    low = expr.lower()

    for op_name, op_type in [("union all", "UNION"), ("union", "UNION"), ("intersect", "INTERSECT"), ("except", "EXCEPT"), ("difference", "EXCEPT")]:
        if low.startswith(op_name + "("):
            op_len = len(op_name)
            if not expr.endswith(")"): raise ValueError(f"❌ Missing closing parenthesis for {op_name}(...).")
            content = expr[op_len+1:-1].strip()
            split_i = _find_top_level_comma(content)
            if split_i != -1:
                left_expr = content[:split_i].strip()
                right_expr = content[split_i+1:].strip()
                is_all = (op_name == "union all")
                return SetOperation(_parse_to_ast(left_expr), _parse_to_ast(right_expr), op_type, is_all)
            raise ValueError(f"❌ Invalid syntax for {op_name}(A, B).")

    depth = 0
    for i in range(len(expr) - 1, -1, -1):
        ch = expr[i]
        if ch == ')': depth += 1
        elif ch == '(': depth -= 1
        if depth == 0:
            for op_name, op_type in [("union all", "UNION"), ("union", "UNION"), ("intersect", "INTERSECT"), ("except", "EXCEPT"), ("difference", "EXCEPT")]:
                op_len = len(op_name)
                if i > op_len and expr[i-op_len+1:i+1].lower() == op_name:
                    if (expr[i-op_len].isspace() and expr[i+1].isspace()):
                        left_expr = expr[:i-op_len].strip()
                        right_expr = expr[i+2:].strip()
                        is_all = (op_name == "union all")
                        return SetOperation(_parse_to_ast(left_expr), _parse_to_ast(right_expr), op_type, is_all)

    if low.startswith("sigma"):
        m = re.match(r"(?i)sigma\s*\[(.*?)?\]\s*\((.*)\)$", expr, re.DOTALL)
        if not m: raise ValueError("❌ Invalid syntax for sigma (use: sigma[condition](Relation)).")
        cond = m.group(1).strip() if m.group(1) else "1=1"
        inner_expr = m.group(2).strip()
        return Select(_parse_to_ast(inner_expr), cond)

    if low.startswith("pi"):
        m = re.match(r"(?i)pi\s*\[(.*?)?\]\s*\((.*)\)$", expr, re.DOTALL)
        if not m: raise ValueError("❌ Invalid syntax for pi (use: pi[attr1,attr2](Relation)).")
        cols_str = m.group(1).strip() if m.group(1) else "*"
        columns = [col.strip() for col in cols_str.split(',')]
        inner_expr = m.group(2).strip()
        return Project(_parse_to_ast(inner_expr), columns)

    if low.startswith("rho") or low.startswith("rename"):
        m = re.match(r"(?i)(?:rho|rename)\s*\[\s*(\w+)(?:,.*)?\s*\]\s*\((.*)\)$", expr, re.DOTALL)
        if not m: raise ValueError("❌ Invalid syntax for rename (use: rho[NewName](Relation)).")
        new_name = m.group(1).strip()
        inner_expr = m.group(2).strip()
        child_node = _parse_to_ast(inner_expr)
        if isinstance(child_node, TableScan) and child_node.alias is None:
             child_node.alias = new_name
             return child_node
        return Rename(child_node, new_name)

    if low.startswith("gamma"):
        m = re.match(r"(?i)gamma\s*\[(.*?)(?:;(.*?))?\]\s*\((.*)\)$", expr, re.DOTALL)
        if not m: raise ValueError("❌ Invalid syntax for gamma (use: gamma[group_by; agg1, agg2](Relation)).")
        group_by_str = m.group(1).strip()
        agg_str = m.group(2).strip() if m.group(2) else ""
        inner_expr = m.group(3).strip()
        group_by_cols = [col.strip() for col in group_by_str.split(',')] if group_by_str else []
        aggregates = [agg.strip() for agg in agg_str.split(',')] if agg_str else []
        if not group_by_cols and not aggregates: aggregates = ["COUNT(*)"]
        elif group_by_cols and not aggregates: aggregates = group_by_cols
        return Aggregate(_parse_to_ast(inner_expr), group_by_cols, aggregates)

    if low.startswith("join"):
        m_cond = re.match(r"(?i)join\s*\[(.*?)?\]\s*\((.*)\)$", expr, re.DOTALL)
        if m_cond:
            cond = m_cond.group(1).strip() if m_cond.group(1) else None
            content = m_cond.group(2).strip()
            split_i = _find_top_level_comma(content)
            if split_i != -1:
                left_expr = content[:split_i].strip()
                right_expr = content[split_i+1:].strip()
                return Join(_parse_to_ast(left_expr), _parse_to_ast(right_expr), cond, join_type='INNER')
            raise ValueError("❌ Invalid syntax for join[...](A, B).")

        m_no_cond = re.match(r"(?i)join\s*\((.*)\)$", expr, re.DOTALL)
        if m_no_cond:
            content = m_no_cond.group(1).strip()
            split_i = _find_top_level_comma(content)
            if split_i != -1:
                left_expr = content[:split_i].strip()
                right_expr = content[split_i+1:].strip()
                return Join(_parse_to_ast(left_expr), _parse_to_ast(right_expr), None, join_type='CROSS')
            raise ValueError("❌ Invalid syntax for join(A, B).")
        raise ValueError("❌ Invalid syntax for join.")

    depth = 0
    for i, ch in enumerate(expr):
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if depth == 0 and ch.lower() == 'x':
            if i > 0 and expr[i - 1].isspace() and i + 1 < len(expr) and expr[i + 1].isspace():
                left_expr = expr[:i].strip()
                right_expr = expr[i + 1:].strip()
                return Join(_parse_to_ast(left_expr), _parse_to_ast(right_expr), None, join_type='CROSS')

    if re.match(r"^[\w]+$", expr):
        return TableScan(expr)
    raise ValueError(f"❌ Invalid or unsupported RA expression: '{expr}'")

# ---------- Phase 3: The "Smart" Compiler (AST -> SQL) ----------
class Query:
    def __init__(self):
        self.select: List[str] = ["*"]
        self.from_table: Optional[str] = None
        self.from_alias: Optional[str] = None
        self.joins: List[str] = []
        self.where: List[str] = []
        self.group_by: List[str] = []
        self.having: List[str] = []
        self.is_complex: bool = False

def _build_sql_string(q: Query) -> str:
    if q.from_table is None:
        if q.joins:
             return "\n".join(filter(None, [f"SELECT {', '.join(q.select)}", "FROM " + q.joins[0], "\n".join(q.joins[1:]), f"WHERE {' AND '.join(q.where)}" if q.where else "", f"GROUP BY {', '.join(q.group_by)}" if q.group_by else "", f"HAVING {' AND '.join(q.having)}" if q.having else ""]))
        raise ValueError("Cannot build SQL query with no FROM clause.")
    from_clause = q.from_table
    if q.from_alias: from_clause += f" AS {q.from_alias}"
    select_s = f"SELECT {', '.join(q.select)}"
    from_s = f"FROM {from_clause}"
    joins_s = "\n".join(q.joins) if q.joins else ""
    where_s = f"WHERE {' AND '.join(q.where)}" if q.where else ""
    group_by_s = f"GROUP BY {', '.join(q.group_by)}" if q.group_by else ""
    having_s = f"HAVING {' AND '.join(q.having)}" if q.having else ""
    return "\n".join(filter(None, [select_s, from_s, joins_s, where_s, group_by_s, having_s]))

def _compile_ast(node: Node):
    if isinstance(node, TableScan):
        q = Query()
        q.from_table = node.name
        if node.alias: q.from_alias = node.alias
        return q
    if isinstance(node, SetOperation):
        left_compiled = _compile_ast(node.left)
        right_compiled = _compile_ast(node.right)
        left_sql = _build_sql_string(left_compiled) if isinstance(left_compiled, Query) else left_compiled
        right_sql = _build_sql_string(right_compiled) if isinstance(right_compiled, Query) else right_compiled
        op_sql = "UNION ALL" if node.op_type == "UNION" and node.is_all else node.op_type
        return f"({left_sql})\n{op_sql}\n({right_sql})"
    if isinstance(node, Select):
        child_result = _compile_ast(node.child)
        if isinstance(child_result, str):
            q = Query()
            q.from_table = f"({child_result}) AS T_Sub"
            q.where = [node.predicate]
            return q
        q = child_result
        if q.is_complex and isinstance(node.child, Aggregate):
            q.having.append(node.predicate)
            return q
        elif q.is_complex:
            inner_sql = _build_sql_string(q)
            new_q = Query()
            new_q.from_table = f"({inner_sql}) AS T_Sub"
            new_q.where = [node.predicate]
            return new_q
        else:
            q.where.append(node.predicate)
            return q
    if isinstance(node, Project):
        child_result = _compile_ast(node.child)
        if isinstance(child_result, str):
            q = Query()
            q.from_table = f"({child_result}) AS T_Sub"
            q.select = node.columns
            return q
        q = child_result
        if q.is_complex:
            inner_sql = _build_sql_string(q)
            new_q = Query()
            new_q.from_table = f"({inner_sql}) AS T_Sub"
            new_q.select = node.columns
            return new_q
        else:
            q.select = node.columns
            return q
    if isinstance(node, Aggregate):
        child_result = _compile_ast(node.child)
        if isinstance(child_result, str):
            q = Query()
            q.from_table = f"({child_result}) AS T_Sub"
        else:
            q = child_result
            if q.is_complex:
                 inner_sql = _build_sql_string(q)
                 q = Query()
                 q.from_table = f"({inner_sql}) AS T_Sub"
        q.group_by = node.group_by_cols
        select_list = []
        if q.group_by: select_list.extend(node.group_by_cols)
        if node.aggregates: select_list.extend(node.aggregates)
        q.select = sorted(list(set(select_list)), key=select_list.index) if select_list else ["*"]
        q.is_complex = True
        return q
    if isinstance(node, Rename):
        child_result = _compile_ast(node.child)
        if isinstance(child_result, str):
            q = Query()
            q.from_table = f"({child_result}) AS {node.new_name}"
            return q
        q = child_result
        if (not q.joins and not q.where and not q.group_by and not q.having and
            q.select == ["*"] and q.from_table is not None and not q.from_alias):
            q.from_alias = node.new_name
            return q
        else:
            inner_sql = _build_sql_string(q)
            new_q = Query()
            new_q.from_table = f"({inner_sql}) AS {node.new_name}"
            return new_q
    if isinstance(node, Join):
        left_result = _compile_ast(node.left)
        right_result = _compile_ast(node.right)
        if isinstance(left_result, str):
            left_q = Query()
            left_q.from_table = f"({left_result}) AS T_Left"
        else:
            left_q = left_result
            if left_q.is_complex or left_q.joins:
                left_sql = _build_sql_string(left_q)
                left_q = Query()
                left_q.from_table = f"({left_sql}) AS T_Left_Sub"
        right_from_sql = ""
        if isinstance(right_result, str):
            right_from_sql = f"({right_result}) AS T_Right"
        else:
            right_q = right_result
            if right_q.is_complex:
                right_from_sql = f"({_build_sql_string(right_q)}) AS T_Right_Sub"
            else:
                right_from_sql = right_q.from_table
                if right_q.from_alias: right_from_sql += f" AS {right_q.from_alias}"
                left_q.joins.extend(right_q.joins)
                left_q.where.extend(right_q.where)
        join_op_str = "CROSS JOIN" if node.join_type == 'CROSS' else "JOIN"
        on_cond = f"ON {node.condition}" if node.condition and node.join_type != 'CROSS' else ""
        left_q.joins.append(f"{join_op_str} {right_from_sql} {on_cond}".strip())
        return left_q
    raise NotImplementedError(f"Unsupported AST node: {type(node)}")

def parse_RA(expr: str) -> str:
    if not expr: return ""
    try:
        ast = _parse_to_ast(expr)
        compiled_result = _compile_ast(ast)
        if isinstance(compiled_result, Query):
            return _build_sql_string(compiled_result)
        else:
            return compiled_result
    except (ValueError, NotImplementedError) as e:
        raise Exception(str(e))
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
# ▲▲▲ YOUR ENTIRE BACKEND CODE (UNCHANGED) ▲▲▲


# --- NEW FLASK API ENDPOINT ---
@app.route('/convert', methods=['POST'])
def convert_api():
    """
    This is the API endpoint that the JavaScript frontend will call.
    It receives the RA expression, calls your parse_RA function,
    and returns the SQL query or an error.
    """
    try:
        data = request.json
        if not data or 'expression' not in data:
            return jsonify({'error': 'No expression provided.'}), 400
        
        expression = data['expression']
        
        # Call your existing, unchanged function!
        sql_query = parse_RA(expression)
        
        return jsonify({'sql': sql_query})
    
    except Exception as e:
        # Send a proper error message back to the frontend
        return jsonify({'error': str(e)}), 400

# --- Run the Server ---
if __name__ == '__main__':
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)