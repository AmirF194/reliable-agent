"""Tool registry with schema validation.

Tool inputs are validated against their JSON schema *before* the tool runs. A bad call
returns a structured error to the model (which can self-correct) instead of crashing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .backends.base import Tool


@dataclass
class RegisteredTool:
    spec: Tool
    fn: Callable[..., str]
    _validator: Draft202012Validator

    def run(self, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Returns (content, is_error). Validation failures are errors, not exceptions."""
        errors = sorted(self._validator.iter_errors(arguments), key=lambda e: e.path)
        if errors:
            msg = "; ".join(f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errors)
            return (f"Invalid arguments: {msg}", True)
        try:
            return (str(self.fn(**arguments)), False)
        except Exception as exc:  # tool bugs become structured errors, never crash the loop
            return (f"Tool '{self.spec.name}' raised {type(exc).__name__}: {exc}", True)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, description: str, input_schema: dict, fn: Callable[..., str]) -> None:
        Draft202012Validator.check_schema(input_schema)
        spec = Tool(name=name, description=description, input_schema=input_schema)
        self._tools[name] = RegisteredTool(spec, fn, Draft202012Validator(input_schema))

    def specs(self) -> list[Tool]:
        return [t.spec for t in self._tools.values()]

    def dispatch(self, name: str, arguments: dict) -> tuple[str, bool]:
        if name not in self._tools:
            return (f"Unknown tool: {name}", True)
        return self._tools[name].run(arguments)


def default_registry() -> ToolRegistry:
    """A couple of safe example tools so the agent does something real out of the box."""
    reg = ToolRegistry()

    def calculator(expression: str) -> str:
        import ast
        import operator as op

        ops = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
            ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg,
        }

        def _eval(node):  # safe arithmetic only — no eval()
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp):
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](_eval(node.operand))
            raise ValueError("unsupported expression")

        return str(_eval(ast.parse(expression, mode="eval").body))

    reg.register(
        "calculator",
        "Evaluate a basic arithmetic expression (+ - * / ** %).",
        {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        calculator,
    )
    return reg
