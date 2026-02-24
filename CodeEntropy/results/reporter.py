"""
Utilities for logging entropy results and exporting data.

This module provides the ResultsReporter class, which is responsible for:

- Collecting molecule-level entropy results
- Collecting residue-level entropy results
- Storing group metadata labels
- Rendering rich tables to the console
- Exporting results to JSON
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from CodeEntropy.core.logging import LoggingConfig

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class ResultsReporter:
    """Collect, format, and output entropy calculation results."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console: Console = console or Console()
        self.molecule_data: List[Tuple[Any, Any, Any, Any]] = []
        self.residue_data: List[List[Any]] = []
        self.group_labels: Dict[Any, Dict[str, Any]] = {}

    @staticmethod
    def clean_residue_name(resname: Any) -> str:
        """Clean residue name by removing dash-like characters."""
        return re.sub(r"[-–—]", "", str(resname))

    @staticmethod
    def _gid_sort_key(x: Any) -> Tuple[int, Any]:
        """
        Stable sort key for group IDs that may be numeric strings, ints, or other
        objects.

        Returns (rank, value):
          - numeric IDs -> (0, int_value)
          - non-numeric -> (1, str_value)
        """
        sx = str(x)
        try:
            return (0, int(sx))
        except Exception:
            return (1, sx)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert value to float if possible; otherwise return None."""
        try:
            if isinstance(value, bool):
                return None
            return float(value)
        except Exception:
            return None

    def add_results_data(
        self, group_id: Any, level: str, entropy_type: str, value: Any
    ) -> None:
        """Add molecule-level entropy result."""
        self.molecule_data.append((group_id, level, entropy_type, value))

    def add_residue_data(
        self,
        group_id: Any,
        resname: str,
        level: str,
        entropy_type: str,
        frame_count: Any,
        value: Any,
    ) -> None:
        """Add residue-level entropy result."""
        resname = self.clean_residue_name(resname)
        if isinstance(frame_count, np.ndarray):
            frame_count = frame_count.tolist()
        self.residue_data.append(
            [group_id, resname, level, entropy_type, frame_count, value]
        )

    def add_group_label(
        self,
        group_id: Any,
        label: str,
        residue_count: Optional[int] = None,
        atom_count: Optional[int] = None,
    ) -> None:
        """Store metadata label for a group."""
        self.group_labels[group_id] = {
            "label": label,
            "residue_count": residue_count,
            "atom_count": atom_count,
        }

    def log_tables(self) -> None:
        """Render all collected data as Rich tables (grouped by group id)."""
        self._log_grouped_results_tables()
        self._log_residue_table_grouped()
        self._log_group_label_table()

    def _log_grouped_results_tables(self) -> None:
        """
        Print molecule-level results grouped by group_id with components + total
        together.
        """
        if not self.molecule_data:
            return

        grouped: Dict[Any, List[Tuple[Any, Any, Any, Any]]] = {}
        for row in self.molecule_data:
            gid = row[0]
            grouped.setdefault(gid, []).append(row)

        for gid in sorted(grouped.keys(), key=self._gid_sort_key):
            label = self.group_labels.get(gid, {}).get("label", "")
            title = f"Entropy Results — Group {gid}" + (f" ({label})" if label else "")

            table = Table(title=title, show_lines=True, expand=True)
            table.add_column("Level", justify="center", style="magenta")
            table.add_column("Type", justify="center", style="green")
            table.add_column("Result (J/mol/K)", justify="center", style="yellow")

            rows = grouped[gid]
            non_total: List[Tuple[str, str, Any]] = []
            totals: List[Tuple[str, str, Any]] = []

            for _gid, level, typ, val in rows:
                level_s = str(level)
                typ_s = str(typ)
                is_total = level_s.lower().startswith(
                    "group total"
                ) or typ_s.lower().startswith("group total")
                if is_total:
                    totals.append((level_s, typ_s, val))
                else:
                    non_total.append((level_s, typ_s, val))

            for level_s, typ_s, val in sorted(non_total, key=lambda r: (r[0], r[1])):
                table.add_row(level_s, typ_s, str(val))

            for level_s, typ_s, val in totals:
                table.add_row(level_s, typ_s, str(val))

            console.print(table)

    def _log_residue_table_grouped(self) -> None:
        """Render residue entropy table grouped by group id."""
        if not self.residue_data:
            return

        grouped: Dict[Any, List[List[Any]]] = {}
        for row in self.residue_data:
            gid = row[0]
            grouped.setdefault(gid, []).append(row)

        for gid in sorted(grouped.keys(), key=self._gid_sort_key):
            label = self.group_labels.get(gid, {}).get("label", "")
            title = f"Residue Entropy — Group {gid}" + (f" ({label})" if label else "")

            table = Table(title=title, show_lines=True, expand=True)
            table.add_column("Residue Name", justify="center", style="cyan")
            table.add_column("Level", justify="center", style="magenta")
            table.add_column("Type", justify="center", style="green")
            table.add_column("Count", justify="center", style="green")
            table.add_column("Result (J/mol/K)", justify="center", style="yellow")

            for _gid, resname, level, typ, count, val in grouped[gid]:
                table.add_row(str(resname), str(level), str(typ), str(count), str(val))

            console.print(table)

    def _log_group_label_table(self) -> None:
        """Render group label metadata table."""
        if not self.group_labels:
            return

        table = Table(title="Group Metadata", show_lines=True, expand=True)
        table.add_column("Group ID", justify="center", style="bold cyan")
        table.add_column("Label", justify="center", style="green")
        table.add_column("Residue Count", justify="center", style="magenta")
        table.add_column("Atom Count", justify="center", style="yellow")

        for group_id in sorted(self.group_labels.keys(), key=self._gid_sort_key):
            info = self.group_labels[group_id]
            table.add_row(
                str(group_id),
                str(info.get("label", "")),
                str(info.get("residue_count", "")),
                str(info.get("atom_count", "")),
            )

        console.print(table)

    def save_dataframes_as_json(
        self,
        molecule_df,
        residue_df,
        output_file: str,
        *,
        args: Optional[Any] = None,
        include_raw_tables: bool = False,
    ) -> None:
        """
        Save results to a grouped JSON structure.

        JSON contains:
          - args: arguments used (serialized)
          - provenance: version, python, platform, optional git sha
          - groups: { "<gid>": { components: {...}, total: ... } }

        Args:
            molecule_df: Pandas DataFrame containing molecule results.
            residue_df: Pandas DataFrame containing residue results.
            output_file: Path to JSON output file.
            args: Optional argparse Namespace or dict of arguments used.
            include_raw_tables: If True, also include old "molecule_data"/"residue_data"
                arrays for debugging/backwards-compat.
        """
        payload = self._build_grouped_payload(
            molecule_df=molecule_df,
            residue_df=residue_df,
            args=args,
            include_raw_tables=include_raw_tables,
        )

        with open(output_file, "w") as out:
            json.dump(payload, out, indent=2)

    def _build_grouped_payload(
        self,
        *,
        molecule_df,
        residue_df,
        args: Optional[Any],
        include_raw_tables: bool,
    ) -> Dict[str, Any]:
        mol_rows = molecule_df.to_dict(orient="records")
        res_rows = residue_df.to_dict(orient="records")

        groups: Dict[str, Dict[str, Any]] = {}

        for row in mol_rows:
            gid = str(row.get("Group ID"))
            level = str(row.get("Level"))
            typ = str(row.get("Type"))

            raw_val = row.get("Result (J/mol/K)")
            val = self._safe_float(raw_val)
            if val is None:
                continue

            groups.setdefault(gid, {"components": {}, "total": None})

            is_total = level.lower().startswith(
                "group total"
            ) or typ.lower().startswith("group total")
            if is_total:
                groups[gid]["total"] = val
            else:
                key = f"{level}:{typ}"
                groups[gid]["components"][key] = val

        for _gid, g in groups.items():
            if g["total"] is None:
                comps = g["components"].values()
                g["total"] = float(sum(comps)) if comps else 0.0

        payload: Dict[str, Any] = {
            "args": self._serialize_args(args),
            "provenance": self._provenance(),
            "groups": groups,
        }

        if include_raw_tables:
            payload["molecule_data"] = mol_rows
            payload["residue_data"] = res_rows

        return payload

    @staticmethod
    def _serialize_args(args: Optional[Any]) -> Dict[str, Any]:
        """Turn argparse Namespace / dict / object into a JSON-serializable dict."""
        if args is None:
            return {}

        if isinstance(args, dict):
            base = dict(args)
        else:
            base = getattr(args, "__dict__", None)
            if not base:
                try:
                    base = dict(args)
                except Exception:
                    return {}

        out: Dict[str, Any] = {}
        for k, v in base.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _provenance() -> Dict[str, Any]:
        prov: Dict[str, Any] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }

        try:
            from importlib.metadata import version

            prov["codeentropy_version"] = version("CodeEntropy")
        except Exception:
            prov["codeentropy_version"] = None

        prov["git_sha"] = ResultsReporter._try_get_git_sha()
        return prov

    @staticmethod
    def _try_get_git_sha() -> Optional[str]:
        env_sha = os.environ.get("CODEENTROPY_GIT_SHA")
        if env_sha:
            return env_sha

        try:
            here = Path(__file__).resolve()
            repo_guess = here.parents[2]

            if not (repo_guess / ".git").exists():
                for p in here.parents:
                    if (p / ".git").exists():
                        repo_guess = p
                        break
                else:
                    return None

            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_guess),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                return None
            sha = (proc.stdout or "").strip()
            return sha or None
        except Exception:
            return None
