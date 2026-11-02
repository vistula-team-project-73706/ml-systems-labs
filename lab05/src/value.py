"""
Lab 05: Scalar automatic differentiation engine

Value is one node of a computation graph. It stores:
    data    the number computed in the forward pass
    grad    d(output)/d(this node), filled in by backward()
    parents the nodes this one was computed from
    rule    the local derivative rule of the operation that made this node

Every operation returns a new Value that points back at its inputs, so the
graph is built as a side effect of the forward pass. backward() then walks
the graph in reverse topological order and applies the chain rule.

Primitive operations: +, *, ** (constant exponent), relu, sum, sigmoid, log.
Derived from those:   neg, -, /

The rule functions take the output node and return one local derivative per
parent, d(out)/d(parent). The chain rule itself lives only in backward().
"""

import math


# ============================================================
# 1. LOCAL DERIVATIVE RULES
# ============================================================

def _add_rule(out):
    return 1.0, 1.0


def _mul_rule(out):
    a, b = out.parents
    return b.data, a.data


def _pow_rule(exponent):
    """Build the rule for a ** exponent, where exponent is a constant."""

    def rule(out):
        # Without this check, 0 ** 0 would divide by zero in 0 ** (0 - 1).
        if exponent == 0:
            return (0.0,)
        return (exponent * out.parents[0].data ** (exponent - 1),)

    return rule


def _relu_rule(out):
    # The derivative at exactly 0 is taken as 0, the same choice as PyTorch.
    return (1.0 if out.parents[0].data > 0 else 0.0,)


def _sum_rule(out):
    return (1.0,)


def _sigmoid_rule(out):
    return (out.data * (1.0 - out.data),)


def _log_rule(out):
    return (1.0 / out.parents[0].data,)


# ============================================================
# 2. VALUE NODE
# ============================================================

class Value:
    """A scalar that remembers how it was computed."""

    # __slots__ because training creates a very large number of these nodes.
    __slots__ = ("data", "grad", "parents", "op", "rule")

    def __init__(self, data, parents=(), op="", rule=None):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.op = op
        self.rule = rule

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self.op!r})"

    # --------------------------------------------------------
    # Primitive operations. Each one adds a node to the graph.
    # --------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+", _add_rule)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*", _mul_rule)

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("Value only supports a constant int or float exponent")
        if self.data < 0 and not float(exponent).is_integer():
            raise ValueError("a negative base needs an integer exponent")
        return Value(self.data ** exponent, (self,), f"**{exponent}", _pow_rule(exponent))

    def relu(self):
        return Value(max(self.data, 0.0), (self,), "relu", _relu_rule)

    def sum(self):
        # A Value is already a scalar. This matches tensor.sum() on a scalar
        # tensor, so the same expression runs on Value and on torch.Tensor.
        return Value(self.data, (self,), "sum", _sum_rule)

    def sigmoid(self):
        # The two forms avoid overflow in exp for large negative or positive x.
        x = self.data
        if x >= 0:
            s = 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            s = e / (1.0 + e)
        return Value(s, (self,), "sigmoid", _sigmoid_rule)

    def log(self):
        if self.data <= 0:
            raise ValueError("log needs a positive input")
        return Value(math.log(self.data), (self,), "log", _log_rule)

    # --------------------------------------------------------
    # Derived operations. These need no derivative rule of their own.
    # --------------------------------------------------------

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __rtruediv__(self, other):
        return Value(other) * self ** -1

    # --------------------------------------------------------
    # Backward pass
    # --------------------------------------------------------

    def topological_order(self):
        """Return every node of the graph, parents before children.

        This is a depth-first post-order walk. It uses an explicit stack so a
        very deep graph cannot hit Python's recursion limit.
        """

        order = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, parents_done = stack.pop()

            if parents_done:
                order.append(node)
                continue

            if node in visited:
                continue

            visited.add(node)
            stack.append((node, True))

            for parent in node.parents:
                if parent not in visited:
                    stack.append((parent, False))

        return order

    def backward(self):
        """Fill in d(self)/d(node).grad for every node in the graph."""

        order = self.topological_order()

        # Interior nodes hold results of an earlier backward() call. If they
        # were kept, a second call would push them into the leaves again.
        # Leaf gradients stay, so they accumulate as they do in PyTorch.
        for node in order:
            if node.parents:
                node.grad = 0.0

        self.grad = 1.0

        for node in reversed(order):
            if node.rule is None:
                continue
            for parent, local_derivative in zip(node.parents, node.rule(node)):
                # += and not =, a node can feed more than one child.
                parent.grad += local_derivative * node.grad
