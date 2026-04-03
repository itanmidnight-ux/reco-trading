from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any


class LLMLogAnalyzer:
    """Analyzes bot logs for errors and writes docs/analisis.txt with enumerated errors."""

    _PROMPT_TEMPLATE = (
        "INSTRUCCIONES PARA LA IA (opencode):\n"
        "=====================================\n"
        "Este archivo contiene errores detectados automaticamente por el LLM de analisis del programa Reco-Trading.\n"
        "Cada vez que recibas este archivo, debes:\n"
        "1. Leer cada error enumerado que NO este tachado (~~RESUELTO~~)\n"
        "2. Analizar el codigo fuente del proyecto para encontrar la causa raiz\n"
        "3. Aplicar las correcciones necesarias sin romper el resto del programa\n"
        "4. Marcar cada error resuelto tachandolo: ~~RESUELTO~~ al final de la linea del error\n"
        "5. NO repetir correcciones de errores ya marcados como ~~RESUELTO~~\n"
        "6. Verificar que las correcciones funcionen correctamente\n"
        "7. Si un error no se puede resolver, agregar una nota explicando por que\n"
        "\n"
        "IMPORTANTE: Despues de resolver los errores, este archivo se limpiara cada 24 horas.\n"
        "Los errores resueltos seran eliminados y se comenzara una nueva lista.\n"
        "=====================================\n\n"
    )

    _ERROR_PATTERNS = [
        (r"(?i)error[:\s]", "ERROR"),
        (r"(?i)exception[:\s]", "EXCEPTION"),
        (r"(?i)traceback", "TRACEBACK"),
        (r"(?i)failed[:\s]", "FAILED"),
        (r"(?i)crash", "CRASH"),
        (r"(?i)no attribute", "MISSING_ATTRIBUTE"),
        (r"(?i)not found", "NOT_FOUND"),
        (r"(?i)cannot assign", "TYPE_ERROR"),
        (r"(?i)key error", "KEY_ERROR"),
        (r"(?i)index out of range", "INDEX_ERROR"),
        (r"(?i)timeout", "TIMEOUT"),
        (r"(?i)connection refused", "CONNECTION_ERROR"),
        (r"(?i)authentication failed", "AUTH_ERROR"),
        (r"(?i)memory", "MEMORY_ISSUE"),
        (r"(?i)leak", "LEAK"),
        (r"(?i)warning[:\s].*critical", "CRITICAL_WARNING"),
    ]

    def __init__(self, docs_dir: str | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.docs_dir = docs_dir or os.path.join(project_root, "docs")
        self.analisis_path = os.path.join(self.docs_dir, "analisis.txt")
        self._error_counter = 0
        self._last_cleanup = datetime.now(timezone.utc)
        existing_errors = self._read_existing_errors()
        if existing_errors:
            self._error_counter = max((e["number"] for e in existing_errors), default=0)

    def _ensure_docs_dir(self) -> None:
        os.makedirs(self.docs_dir, exist_ok=True)

    def _read_existing_errors(self) -> list[dict[str, Any]]:
        """Read existing errors from analisis.txt."""
        if not os.path.exists(self.analisis_path):
            return []

        errors = []
        with open(self.analisis_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse numbered errors
        pattern = r"(\d+)\.\s+\[(RESUELTO|PENDIENTE)\]\s+(.+?)(?=\n\d+\.|\n\n|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for num, status, desc in matches:
            errors.append({
                "number": int(num),
                "status": status,
                "description": desc.strip(),
            })

        return errors

    def _write_analisis_file(self, errors: list[dict[str, Any]]) -> None:
        """Write the analisis.txt file with all errors."""
        self._ensure_docs_dir()

        with open(self.analisis_path, "w", encoding="utf-8") as f:
            f.write(self._PROMPT_TEMPLATE)
            f.write(f"Archivo generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
            f.write(f"Total de errores: {len(errors)}\n")
            f.write(f"Pendientes: {sum(1 for e in errors if e['status'] == 'PENDIENTE')}\n")
            f.write(f"Resueltos: {sum(1 for e in errors if e['status'] == 'RESUELTO')}\n")
            f.write("=" * 60 + "\n\n")

            for error in errors:
                status_mark = "RESUELTO" if error["status"] == "RESUELTO" else "PENDIENTE"
                f.write(f"{error['number']}. [{status_mark}] {error['description']}\n\n")

            if not errors:
                f.write("No hay errores pendientes. El programa funciona correctamente.\n")

    def analyze_logs(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyze log entries and extract errors."""
        existing_errors = self._read_existing_errors()
        self._error_counter = max((e["number"] for e in existing_errors), default=0)
        existing_descriptions = {e["description"] for e in existing_errors}
        new_errors = []

        for entry in logs:
            message = str(entry.get("message", ""))
            level = str(entry.get("level", "")).upper()

            if level not in ("ERROR", "WARNING", "CRITICAL"):
                continue

            # Check against error patterns
            for pattern, error_type in self._ERROR_PATTERNS:
                if re.search(pattern, message):
                    # Normalize the error description to avoid duplicates
                    normalized_desc = self._normalize_error(message, error_type)

                    if normalized_desc not in existing_descriptions:
                        self._error_counter += 1
                        new_error = {
                            "number": self._error_counter,
                            "status": "PENDIENTE",
                            "description": normalized_desc,
                        }
                        new_errors.append(new_error)
                        existing_descriptions.add(normalized_desc)
                    break

        return new_errors

    def _normalize_error(self, message: str, error_type: str) -> str:
        """Normalize error message for deduplication."""
        # Remove timestamps and variable parts
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "", message)
        normalized = re.sub(r"0x[0-9a-fA-F]+", "0xHEX", normalized)
        normalized = re.sub(r"/[^\s]+/", "/path/", normalized)
        normalized = normalized.strip()

        return f"[{error_type}] {normalized[:200]}"

    def add_error(self, description: str, error_type: str = "MANUAL") -> dict[str, Any]:
        """Manually add an error to the analysis file."""
        existing_errors = self._read_existing_errors()
        normalized_desc = self._normalize_error(description, error_type)

        # Check if already exists
        for error in existing_errors:
            if error["description"] == normalized_desc:
                return error

        self._error_counter = max((e["number"] for e in existing_errors), default=0) + 1
        new_error = {
            "number": self._error_counter,
            "status": "PENDIENTE",
            "description": normalized_desc,
        }
        existing_errors.append(new_error)
        self._write_analisis_file(existing_errors)
        self.logger.info(f"New error added to analisis.txt: #{self._error_counter}")
        return new_error

    def mark_resolved(self, error_number: int) -> bool:
        """Mark an error as resolved."""
        existing_errors = self._read_existing_errors()
        for error in existing_errors:
            if error["number"] == error_number:
                error["status"] = "RESUELTO"
                self._write_analisis_file(existing_errors)
                self.logger.info(f"Error #{error_number} marked as resolved")
                return True
        return False

    def cleanup_old_errors(self) -> int:
        """Clean up resolved errors every 24 hours. Returns count of removed errors."""
        now = datetime.now(timezone.utc)
        if (now - self._last_cleanup).total_seconds() < 86400:
            return 0

        existing_errors = self._read_existing_errors()
        resolved_count = sum(1 for e in existing_errors if e["status"] == "RESUELTO")
        pending_errors = [e for e in existing_errors if e["status"] == "PENDIENTE"]

        # Renumber pending errors
        for i, error in enumerate(pending_errors, 1):
            error["number"] = i

        self._error_counter = len(pending_errors)
        self._write_analisis_file(pending_errors)
        self._last_cleanup = now

        self.logger.info(f"Cleanup: removed {resolved_count} resolved errors, {len(pending_errors)} pending")
        return resolved_count

    def get_pending_errors(self) -> list[dict[str, Any]]:
        """Get list of pending (unresolved) errors."""
        existing_errors = self._read_existing_errors()
        return [e for e in existing_errors if e["status"] == "PENDIENTE"]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of error analysis."""
        existing_errors = self._read_existing_errors()
        pending = [e for e in existing_errors if e["status"] == "PENDIENTE"]
        resolved = [e for e in existing_errors if e["status"] == "RESUELTO"]

        return {
            "total_errors": len(existing_errors),
            "pending": len(pending),
            "resolved": len(resolved),
            "pending_errors": pending,
            "last_cleanup": self._last_cleanup.isoformat(),
            "file_exists": os.path.exists(self.analisis_path),
        }
