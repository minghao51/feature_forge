"""Code generation + static validation for MALMAS feature pipelines.

Extracted from ``core.py`` to keep the orchestration god-class focused.
The import policy is owned by :mod:`feature_forge.evaluation.sandbox`
(``BANNED_IMPORTS`` here, ``ALLOWED_IMPORTS`` on :class:`SandboxedExecutor`).
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from typing import Any

from feature_forge.evaluation.sandbox import BANNED_IMPORTS
from feature_forge.exceptions import PipelineError
from feature_forge.llm.base import LLMClient
from feature_forge.methods.malmas.prompts import get_registry
from feature_forge.observability.structlog_config import get_logger
from feature_forge.types import FeatureSpec
from feature_forge.utils import strip_markdown_fences

logger = get_logger(__name__)


class CodeGenerator:
    """Generates pandas code from feature specifications."""

    def __init__(self, llm_client: LLMClient, max_tokens: int = 32768) -> None:
        self.llm_client = llm_client
        self._system_prompt = get_registry().get("code_generation").system
        self._max_tokens = max_tokens

    async def generate_code(
        self,
        specs: list[FeatureSpec],
        schema: dict[str, Any] | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate Python code for a list of feature specs.

        Includes AST validation and a single retry on syntax errors.
        """
        specs_dump = [s.model_dump() if hasattr(s, "model_dump") else s for s in specs]
        specs_json = json.dumps(specs_dump, indent=2, ensure_ascii=False)

        parts: list[str] = []
        if schema:
            schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
            parts.append(f"Data schema:\n{schema_json}")
        parts.append(f"Generate code for features:\n{specs_json}")
        user_prompt = "\n\n".join(parts)

        if error_feedback:
            user_prompt += f"\n\n{error_feedback}"

        last_error: str | None = None
        for attempt in range(2):
            response = await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=self._max_tokens,
            )
            code = strip_markdown_fences(response.content)

            error_msg = _validate_code_ast(code)
            if error_msg is None:
                return code
            last_error = error_msg

            if attempt == 0:
                logger.warning("code_gen_ast_invalid", error=error_msg, attempt=attempt)
                user_prompt = (
                    user_prompt + f"\n\nYour previous code failed with: {error_msg}\n"
                    "Please fix the code and output only valid Python."
                )
            else:
                logger.error("code_gen_ast_retry_failed", error=error_msg)

        raise PipelineError(
            "Generated code failed validation after 2 attempts: "
            f"{last_error or 'unknown validation error'}"
        )


def _validate_code_ast(code: str) -> str | None:
    """Validate generated Python code via AST + linting (no execution).

    Returns None if valid, or an error message string if not.
    Checks:
    - Valid Python syntax (AST parse)
    - Contains generate_features function
    - No banned imports (denylist sourced from :data:`BANNED_IMPORTS`)
    - No undefined names (ruff lint, catches bare column refs like 'f1')
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Python syntax error: {exc}"

    has_generate_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "generate_features"
        for node in ast.walk(tree)
    )
    if not has_generate_fn:
        return "Missing generate_features(df) function definition"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORTS:
                    return f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in BANNED_IMPORTS:
                    return f"Forbidden import: {node.module}"

    if shutil.which("ruff") is None:
        return None

    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F821,F823", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 and result.stdout.strip():
            first_error = (
                result.stdout.strip().split("\n")[0] if result.stdout.strip() else "lint error"
            )
            return f"Undefined name or reference: {first_error}"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None
