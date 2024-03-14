import ast
from dataclasses import dataclass
from functools import reduce
import pprint
import re
from BitVector import BitVector

from nodes import PLNode, PLScopeNode, PLConst, PLBinOp, PLVariable, plnode_walk, PLCall

def iter_fields(node):
    for field in node._fields:
        try:
            yield getattr(node, field), field
        except AttributeError:
            pass

UnrollIndexMap = dict[str, tuple[int, int]]
ResourceCtx = dict[str, list[tuple[int, int]]]

@dataclass(eq=True, frozen=True)
class AccessPattern:
    off: int
    index_map: UnrollIndexMap
    base_exprs: list[PLNode]
    has_effects: bool

AccessMap = dict[str, list[tuple[list[AccessPattern], list[AccessPattern]]]]

class PLAffineTyper:
    def __init__(self, arg_info, debug=False):
        self.arg_info = arg_info
        self.debug = debug
        self.res_ctx: ResourceCtx = {}
        self.unroll_ctx: UnrollIndexMap = {}
        self.access_map: AccessMap = {}

    def visit(self, node):
        """Visit a node in preorder."""

        if node == None:
            return

        if self.debug:
            print(f'Visiting {node.__class__.__name__}, {node}')
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""

        for field, name in iter_fields(node):
            if isinstance(node, PLScopeNode) and name == 'body':
                self.check_and_visit_body(field)
            elif isinstance(field, PLNode) or isinstance(field, list):
                self.visit(field)

    def visit_list(self, l):
        for item in filter(lambda x: isinstance(x, PLNode), l):
            self.visit(item)

    def check_and_visit_body(self, body):
        for stmt in filter(lambda x: isinstance(x, PLNode), body):
            self.visit(stmt)
            if not isinstance(stmt, PLScopeNode):
                self.affine_check(stmt)
                self.access_map = {}

    def affine_check(self, stmt) -> bool:
        if self.debug:
            print("--------- affine checking at simple statement: ", ast.unparse(stmt.ast_node))
            print("resource context", self.res_ctx)
            print("unroll context", self.unroll_ctx)
            print("accesses: ", self.access_map)
        success = True
        for var, accesses in self.access_map.items():
            for i, (writes, reads) in enumerate(accesses):
                if len(writes) == 0:
                    continue
                #TODO: pretty print access patterns 
                first = writes[0]
                res_count = self.res_ctx[var][i][1]

                for access in writes[1:] + reads:
                    if access.base_exprs != first.base_exprs:
                        print(f"Affine Failure: incompatible access patterns {access} and {first} to dimension {i} of {var} on line {stmt.ast_node.lineno}")
                        success = False
                    # check for non-affine policies:
                    for unroll_var, (factor, increment) in access.index_map.items():
                        if not res_count % (factor * inc) == 0:
                            print(f"Affine Failure: unroll factor times increment of unroll variable {unroll_var} does not divide the resource count {res_count} of dimension {i} of {var} on line {stmt.ast_node.lineno}")

                resources = BitVector(size=res_count)
                if first.has_effects:
                    print(f"Affine Warning: function call in {first} on line {stmt.ast_node.lineno} consumes all resources")
                    resources = ~resources

                basePattern = AccessPattern(0, first.index_map, first.base_exprs, False) 
                consumeVec, conf_bit = get_consume_vec(res_count, basePattern) 
                if conf_bit != -1 :
                    print(f"Affine Error: partition {conf_bit} of dimension {i} of {var} consumed more than once by {first} on line {stmt.ast_node.lineno}")
                    success = False

                for read in reads:
                    resources |= consumeVec.deep_copy() >> read.off 
                for write in writes:
                    toConsume = consumeVec.deep_copy() >> write.off
                    conf_bit = (resources & toConsume).next_set_bit() 
                    if conf_bit != -1:
                        print(f"Affine Error: partition {conf_bit} starting from {basePattern} of dimension {i} of {var} accessed by {write} already consumed on line {stmt.ast_node.lineno}")
                        success = False

                    resources |= toConsume

        return success

    def visit_PLCall(self, node):
        if node.is_method:
            self.visit(node.obj)
        self.generic_visit(node)

    def visit_PLSubscript(self, node):
        wr_index = 0 if isinstance(node.ast_node.ctx, ast.Store) else 1
        name = node.var.name
        if name in self.res_ctx:
            # else must be range wise bit-access
            res_dims = len(self.res_ctx[name])
            for i, index in enumerate(node.indices):
                pattern = get_access_pattern(index, self.unroll_ctx)
                if name not in self.access_map:
                    self.access_map[name] =[([], []) for _ in range(res_dims)]
                self.access_map[name][i][wr_index].append(pattern)

        self.generic_visit(node)

    def visit_PLPragma(self, node):
        pragma = node.pragma.value
        m1 = re.search(r'HLS array_partition variable=(\S*)(.*)', pragma)
        if m1 is not None:
            var_name = m1.group(1)
            params = m1.group(2)
            m2 = re.search(' (?:type=)?(cyclic|complete|block)', params)
            type = m2.group(1) if m2 is not None else None
            m2 = re.search(' dim=(\S*)', params)
            dim = int(m2.group(1))if m2 is not None else 0
            m2 = re.search(' factor=(\S*)', params)
            factor = int(m2.group(1)) if m2 is not None else None

            if type == "block" and self.debug:
                print("Affine Failure: block partition not supported on line {node.ast_node.lineno}")
                return
            
            resources = self.res_ctx[var_name]
            if dim > len(resources):
                print(f"Affine Error: parition of {var_name} at nonexistent dimension {dim} on line {node.ast_node.lineno}")
                return 
            idx = dim - 1
            
            if type == "complete":
                if dim == 0:
                    self.res_ctx[var_name] = list(map(lambda pair: (pair[0], pair[0]), resources))
                else:
                    l, _ = resources[idx]
                    resources[idx] = (l, l)

            if type == "cyclic":
                if dim == 0:
                    self.res_ctx[var_name] = list(map(lambda pair: (pair[0], factor), resources))
                else:
                    l, _ = resources[idx]
                    resources[idx] = (l, factor)

            # print(f"array_partition: factor={factor}, dim={dim}, type={type}")
        self.generic_visit(node)

    def visit_PLFor(self, node):
        iter_dom = node.iter_dom
        var_name = None
        shadowed = None
        if iter_dom.attr == 'unroll':
            assert iter_dom.type == 'range'         
            unroll_factor = iter_dom.attr_args[0].value if len(node.iter_dom.attr_args) > 0 else (iter_dom.end.value - iter_dom.start.value) // iter_dom.step.value 
            if unroll_factor > 1:
                var_name = node.target.name

                if var_name in self.unroll_ctx:
                    shadowed = self.unroll_ctx[var_name]
                self.unroll_ctx[var_name] = (unroll_factor, iter_dom.step.value)
        
        self.generic_visit(node)
        if var_name is not None:
            if shadowed is not None:
                self.unroll_ctx[var_name] = shadowed
            else:
                del self.unroll_ctx[var_name]

    def visit_PLFunctionDef(self, node):
        for arg in node.args:
            _, shape = self.arg_info[arg.name]
            self.res_ctx[arg.name] =list(map(lambda l: (l, 1), shape))
        self.generic_visit(node)
        # Maybe clear res_ctx at end of other scope nodes as well?
        self.res_ctx = {}

    def visit_PLAssign(self, node):
        if node.is_decl:
            target = node.target
            self.res_ctx[target.name] =list(map(lambda l: (l, 1), target.pl_shape))
        
        self.generic_visit(node)

    def visit_PLArrayDecl(self, node):
        self.res_ctx[node.name.name] =list(map(lambda l: (l, 1), node.pl_shape))
        
        self.generic_visit(node)

def get_access_pattern(node, unroll_ctx) -> AccessPattern:
    #TODO: a normalize function for the node that does constant propagation

    def is_unroll_index_var(node):
        return isinstance(node, PLVariable) and node.name in unroll_ctx

    off = 0
    index_map = {}
    base_exprs = []
    has_effects = False

    def extract_pattern(node) -> bool:
        if isinstance(node, PLConst):
            nonlocal off
            off += node.value 
        elif is_unroll_index_var(node):
            factor, loop_inc = unroll_ctx[node.name]
            index_map[node.name] = factor, loop_inc
        elif isinstance(node, PLBinOp) and node.op == "*":
            if is_unroll_index_var(node.left) and isinstance(PLConst, node.right):
                name = node.left.name
                mult = node.right.value
                factor, loop_inc = unroll_ctx[name]
                if name in index_map:
                    _, inc = index_map[name]
                    index_map[name] = factor, inc + loop_inc * mult 
                else:
                    index_map[name] = factor, loop_inc * mult
            elif is_unroll_index_var(node.right) and isinstance(PLConst, node.left):
                name = node.right.name
                mult = node.left.value
                if name in index_map:
                    _, inc = index_map[name]
                    index_map[name] = factor, inc + loop_inc * mult 
                else:
                    index_map[name] = factor, loop_inc * mult
            else:
                base_exprs.append(node)
        else:
            base_exprs.append(node)

    def extract_from_add_nodes(node):
        if isinstance(node, PLBinOp) and node.op == "+": 
            extract_from_add_nodes(node.left)
            extract_from_add_nodes(node.right)
        else:
            extract_pattern(node)

    extract_from_add_nodes(node)
    base_exprs.sort(key = lambda x: x.__repr__())

    for expr in base_exprs:
        for node in plnode_walk(expr):
            if isinstance(node, PLVariable) and node.name in unroll_ctx:
                print(f"Affine Failure: unanalyzable {expr} on line {expr.ast_node.lineno} due to unrolled index vairable {node.name}")
            if isinstance(node, PLCall):
                has_effects = True
    return AccessPattern(off, index_map, base_exprs, has_effects)

def get_consume_vec(res_count, pattern: AccessPattern) -> tuple[BitVector, int]:
    acc = BitVector(size=res_count)
    index_vars = list(pattern.index_map.items())
    acc[pattern.off] = 1

    for i, (_, (factor, inc)) in enumerate(pattern.index_map.items()):
        if i == 0 and inc == 1:
            # fast path
            if factor > res_count:
                return ~BitVector(size=factor), 0
            acc[pattern.off:pattern.off+factor] = ~BitVector(size=factor)
        else:
            last_acc = acc.deep_copy()
            for unrolled_iter in range(1, factor):
                shifted = last_acc.deep_copy() >> (unrolled_iter * inc)
                # alternatively store conflicts in a bit vector conf
                # and check at the end - should lead to less checks than
                # if conflicts are rare 
                # conf |= acc & shifted
                conf_bit = (acc & shifted).next_set_bit() 
                if conf_bit != -1:
                    return acc, conf_bit
                acc |= shifted
    return acc, -1

def testGetConsumeVec():
    pattern = AccessPattern(0, {"a": (2, 1), "b":(2, 2) }, None, False)
    print(get_consume_vec(4, pattern))

if __name__ == '__main__':
    testGetConsumeVec()
