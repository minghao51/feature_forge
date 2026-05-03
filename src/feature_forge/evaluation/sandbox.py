"""Sandboxed code execution for LLM-generated feature engineering code.

Uses AST validation + restricted builtins to safely execute untrusted code.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from feature_forge.exceptions import CodeExecutionError


class SandboxedExecutor:
    """AST-validated, restricted-builtin code execution.

    Security model:
    - Parse code into AST and walk it for forbidden nodes
    - Only allow imports of pandas, numpy, math
    - Restrict builtins to a whitelist
    - No file operations, no eval/exec, no dynamic imports
    """

    FORBIDDEN_NAMES: ClassVar[set[str]] = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "exit",
        "quit",
        "__builtins__",
    }

    ALLOWED_BUILTINS: ClassVar[set[str]] = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "round",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "enumerate",
        "filter",
        "set",
        "frozenset",
        "isinstance",
        "hasattr",
        "getattr",
        "type",
        "__import__",
    }

    ALLOWED_IMPORTS: ClassVar[set[str]] = {"pandas", "numpy", "math"}

    def execute(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute feature generation code safely.

        Args:
            code: Python code string containing a `generate_features(df)` function.
            df: Input DataFrame.

        Returns:
            DataFrame of generated features.

        Raises:
            CodeExecutionError: If code is invalid or violates sandbox rules.
        """
        tree = self._parse_and_validate(code)

        # Build restricted globals
        safe_globals: dict[str, Any] = {
            "__builtins__": {name: getattr(builtins, name) for name in self.ALLOWED_BUILTINS},
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
        }
        # Add math module
        import math
        safe_globals["math"] = math

        local_vars: dict[str, Any] = {}
        exec(compile(tree, filename="<sandbox>", mode="exec"), safe_globals, local_vars)

        if "generate_features" not in local_vars:
            raise CodeExecutionError("Code must define a 'generate_features(df)' function")

        try:
            result = local_vars["generate_features"](df)
        except Exception as exc:
            raise CodeExecutionError(f"Feature generation execution failed: {exc}") from exc

        if not isinstance(result, pd.DataFrame):
            raise CodeExecutionError(f"generate_features must return a DataFrame, got {type(result).__name__}")

        return result

    def _parse_and_validate(self, code: str) -> ast.AST:
        """Parse and validate code AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise CodeExecutionError(f"Invalid syntax: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self.ALLOWED_IMPORTS:
                        raise CodeExecutionError(f"Import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root not in self.ALLOWED_IMPORTS:
                    raise CodeExecutionError(f"Import from not allowed: {node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_NAMES:
                    raise CodeExecutionError(f"Forbidden function call: {node.func.id}")
            elif isinstance(node, ast.Name):
                if node.id in self.FORBIDDEN_NAMES and isinstance(node.ctx, ast.Load):
                    raise CodeExecutionError(f"Forbidden name reference: {node.id}")

        return tree
