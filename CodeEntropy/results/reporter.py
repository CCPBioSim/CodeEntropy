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
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from CodeEntropy.core.logging import LoggingConfig

logger = logging.getLogger(__name__)
console = LoggingConfig.get_console()


class _RichProgressSink:
    """Thin wrapper around rich.Progress.

    Keeps Rich usage inside the reporting layer so compute/orchestration code
    can emit progress without importing Rich.
    """

    def __init__(self, progress: Progress):
        """Initialise a progress sink that delegates to a rich.Progress instance.

        Args:
            progress: Rich Progress instance used to create/update/advance tasks.
        """
        self._progress = progress

    def add_task(self, description: str, total: int, **fields):
        """Add a progress task to the underlying rich.Progress instance.

        Args:
            description: Task description shown by Rich.
            total: Total number of steps for the task.
            **fields: Additional Rich task fields (e.g., title).

        Returns:
            The task id returned by rich.Progress.add_task.
        """
        fields.setdefault("title", "")
        return self._progress.add_task(description, total=total, **fields)

    def advance(self, task_id, step: int = 1) -> None:
        """Advance a progress task by a number of steps.

        Args:
            task_id: Rich task identifier.
            step: Number of steps to advance the task by.
        """
        self._progress.advance(task_id, step)

    def update(self, task_id, **fields) -> None:
        """Update fields for an existing progress task.

        Args:
            task_id: Rich task identifier.
            **fields: Task fields to update. If "title" is provided as None, it is
                coerced to an empty string for compatibility with Rich rendering.
        """
        if "title" in fields and fields["title"] is None:
            fields["title"] = ""
        self._progress.update(task_id, **fields)


class ResultsReporter:
    """Collect, format, and output entropy calculation results.

    This reporter accumulates:

    - Molecule-level results (group_id, level, entropy_type, value)
    - Residue-level results (group_id, resname, level, entropy_type, frame_count,
      value)
    - Group metadata labels (label, residue_count, atom_count)

    It can render tables using Rich and export grouped results to JSON with basic
    provenance metadata.

    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialise a ResultsReporter.

        Args:
            console: Optional Rich Console to use for rendering. If None, a default
                Console instance is created.
        """
        self.console: Console = console or Console()
        self.molecule_data: list[tuple[Any, Any, Any, Any]] = []
        self.residue_data: list[list[Any]] = []
        self.group_labels: dict[Any, dict[str, Any]] = {}

    @staticmethod
    def clean_residue_name(resname: Any) -> str:
        """Clean residue name by removing dash-like characters.

        Args:
            resname: Residue name (any type, will be converted to str).

        Returns:
            Residue name with dash-like characters removed.
        """
        return re.sub(r"[-–—]", "", str(resname))

    @staticmethod
    def _gid_sort_key(x: Any) -> tuple[int, Any]:
        """Stable sort key for group IDs.

        Group IDs may be numeric strings, ints, or other objects.

        Returns a tuple (rank, value):
          - numeric IDs -> (0, int_value)
          - non-numeric -> (1, str_value)

        Args:
            x: Group identifier.

        Returns:
            Tuple used as a stable sorting key.
        """
        sx = str(x)
        try:
            return (0, int(sx))
        except Exception:
            return (1, sx)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert value to float if possible; otherwise return None.

        Args:
            value: Value to convert.

        Returns:
            Float representation of value, or None if conversion is not possible or
            value is a boolean.
        """
        try:
            if isinstance(value, bool):
                return None
            return float(value)
        except Exception:
            return None

    def add_results_data(
        self, group_id: Any, level: str, entropy_type: str, value: Any
    ) -> None:
        """Add molecule-level entropy result.

        Args:
            group_id: Group identifier.
            level: Hierarchy level label.
            entropy_type: Entropy component/type label.
            value: Result value to store (kept as-is).
        """
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
        """Add residue-level entropy result.

        Args:
            group_id: Group identifier.
            resname: Residue name (will be cleaned to remove dash-like characters).
            level: Hierarchy level label.
            entropy_type: Entropy component/type label.
            frame_count: Number of frames contributing to the value (may be ndarray).
            value: Result value to store (kept as-is).
        """
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
        residue_count: int | None = None,
        atom_count: int | None = None,
    ) -> None:
        """Store metadata label for a group.

        Args:
            group_id: Group identifier.
            label: Human-readable label for the group.
            residue_count: Optional residue count for the group.
            atom_count: Optional atom count for the group.
        """
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
        """Print molecule-level results grouped by group_id with components + total.

        Results are grouped by group_id and rendered as separate tables per group.
        """
        if not self.molecule_data:
            return

        grouped: dict[Any, list[tuple[Any, Any, Any, Any]]] = {}
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
            non_total: list[tuple[str, str, Any]] = []
            totals: list[tuple[str, str, Any]] = []

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

        grouped: dict[Any, list[list[Any]]] = {}
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
        args: Any | None = None,
        include_raw_tables: bool = False,
    ) -> None:
        """Save results to a grouped JSON structure.

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
        args: Any | None,
        include_raw_tables: bool,
    ) -> dict[str, Any]:
        """Build a grouped JSON-serializable payload from result dataframes.

        Args:
            molecule_df: Pandas DataFrame containing molecule results.
            residue_df: Pandas DataFrame containing residue results.
            args: Optional argparse Namespace or dict of arguments used.
            include_raw_tables: If True, include raw dataframe record arrays in payload.

        Returns:
            Dictionary payload suitable for JSON serialization.
        """
        mol_rows = molecule_df.to_dict(orient="records")
        res_rows = residue_df.to_dict(orient="records")

        groups: dict[str, dict[str, Any]] = {}

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

        payload: dict[str, Any] = {
            "args": self._serialize_args(args),
            "provenance": self._provenance(),
            "groups": groups,
        }

        if include_raw_tables:
            payload["molecule_data"] = mol_rows
            payload["residue_data"] = res_rows

        return payload

    @staticmethod
    def _serialize_args(args: Any | None) -> dict[str, Any]:
        """Turn argparse Namespace / dict / object into a JSON-serializable dict.

        Args:
            args: argparse Namespace, dict, or other object with __dict__/iterable.

        Returns:
            JSON-serializable dict of argument values. Unsupported/unreadable inputs
            return an empty dict.
        """
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

        out: dict[str, Any] = {}
        for k, v in base.items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            elif isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _provenance() -> dict[str, Any]:
        """Build a provenance dictionary for exported results.

        Returns:
            Dictionary with python version, platform string, CodeEntropy package
            version (if available), and git sha (if available).
        """
        prov: dict[str, Any] = {
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
    def _try_get_git_sha() -> str | None:
        """Try to determine the current git commit SHA.

        The SHA is obtained from:
          1) Environment variable CODEENTROPY_GIT_SHA, if set.
          2) A git repository discovered by walking up from this file path and
             running `git rev-parse HEAD`.

        Returns:
            Git SHA string if found, otherwise None.
        """
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
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                return None
            sha = (proc.stdout or "").strip()
            return sha or None
        except Exception:
            return None

    @contextmanager
    def progress(self, *, transient: bool = True):
        """Create a workflow progress context.

        Usage:
            with reporter.progress() as p:
                ...

        Args:
            transient: Whether the progress display should be removed on exit.

        Yields:
            A _RichProgressSink that exposes add_task(), update(), and advance().
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[title]}", justify="right"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=transient,
        )
        with progress:
            yield _RichProgressSink(progress)
