from __future__ import annotations

import json
import os
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TESTDATA_BASE_URL = "https://www.ccpbiosim.ac.uk/file-store/codeentropy-testing"


@dataclass(frozen=True)
class RunResult:
    """Holds outputs and metadata from a single CodeEntropy regression run.

    Attributes:
        workdir: Working directory used to run CodeEntropy.
        job_dir: The most recent job directory created by CodeEntropy.
        output_json: Path to the JSON output produced by CodeEntropy.
        payload: Parsed JSON payload.
        stdout: Captured stdout from the CodeEntropy process.
        stderr: Captured stderr from the CodeEntropy process.
    """

    workdir: Path
    job_dir: Path
    output_json: Path
    payload: dict[str, Any]
    stdout: str
    stderr: str


def _repo_root_from_this_file() -> Path:
    """Return repository root inferred from this file location.

    Returns:
        Repository root path.
    """
    return Path(__file__).resolve().parents[2]


def _testdata_root() -> Path:
    """Return the local on-disk cache root for regression input datasets.

    Returns:
        Path to the local test data cache root.
    """
    return _repo_root_from_this_file() / ".testdata"


def _is_within_directory(base: Path, target: Path) -> bool:
    """Check whether a target path is within a base directory.

    Args:
        base: Base directory.
        target: Target path.

    Returns:
        True if target is within base, otherwise False.
    """
    base = base.resolve()
    try:
        target = target.resolve()
        return str(target).startswith(str(base) + os.sep) or target == base
    except FileNotFoundError:
        return _is_within_directory(base, target.parent)


def _safe_extract_tar_gz(tar_gz: Path, dest_dir: Path) -> None:
    """Extract a .tar.gz file safely into dest_dir.

    This prevents path traversal by validating extracted member paths.

    Args:
        tar_gz: Path to the tar.gz archive.
        dest_dir: Destination directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_gz, "r:gz") as tf:
        for member in tf.getmembers():
            member_path = dest_dir / member.name
            if not _is_within_directory(dest_dir, member_path):
                raise RuntimeError(f"Unsafe path in tarball: {member.name}")
        tf.extractall(dest_dir)


def _download(url: str, dst: Path) -> None:
    """Download a URL to a local file path.

    Args:
        url: Source URL.
        dst: Destination file path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as r, dst.open("wb") as f:
            f.write(r.read())
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def ensure_testdata_for_system(system: str, *, required_paths: list[Path]) -> Path:
    """Ensure the filestore dataset for a system exists locally.

    This downloads and extracts <system>.tar.gz from the CCPBioSim HTTPS filestore
    into <repo_root>/.testdata. The archive is expected to contain a top-level
    '<system>/' directory.

    Args:
        system: System name (e.g., 'methane').
        required_paths: Absolute paths that must exist after extraction.

    Returns:
        Path to the system directory under the local cache.

    Raises:
        RuntimeError: If download/extraction fails or required files remain missing.
    """
    root = _testdata_root()
    system_dir = root / system
    tar_path = root / f"{system}.tar.gz"
    url = f"{DEFAULT_TESTDATA_BASE_URL.rstrip('/')}/{system}.tar.gz"

    def all_required_exist() -> bool:
        return all(p.exists() for p in required_paths)

    if required_paths and all_required_exist():
        return system_dir

    root.mkdir(parents=True, exist_ok=True)
    _download(url, tar_path)

    if system_dir.exists():
        for p in sorted(system_dir.rglob("*"), reverse=True):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            except OSError:
                pass
        try:
            system_dir.rmdir()
        except OSError:
            pass

    _safe_extract_tar_gz(tar_path, root)

    if not system_dir.exists():
        raise RuntimeError(
            f"Extraction did not create expected folder {system_dir}. "
            f"Tarball may not contain '{system}/'. url={url}"
        )

    if required_paths and not all_required_exist():
        found = [
            str(p.relative_to(system_dir)) for p in system_dir.rglob("*") if p.is_file()
        ]
        found.sort()
        raise RuntimeError(
            "Regression data extracted but required files are missing.\n"
            f"system={system}\n"
            f"expected:\n  - " + "\n  - ".join(str(p) for p in required_paths) + "\n"
            f"found in {system_dir}:\n  - "
            + ("\n  - ".join(found) if found else "<no files>")
            + "\n"
            f"url={url}\n"
        )

    return system_dir


def _find_latest_job_dir(workdir: Path) -> Path:
    """Find the most recent CodeEntropy job directory in workdir.

    Args:
        workdir: Working directory.

    Returns:
        Path to the latest job directory.

    Raises:
        FileNotFoundError: If no job directory exists.
    """
    job_dirs = sorted(
        [p for p in workdir.iterdir() if p.is_dir() and p.name.startswith("job")]
    )
    if not job_dirs:
        raise FileNotFoundError(f"No job*** folder created in {workdir}")
    return job_dirs[-1]


def _pick_output_json(job_dir: Path) -> Path:
    """Pick the primary JSON output file from a CodeEntropy job directory.

    Args:
        job_dir: CodeEntropy job directory.

    Returns:
        Path to the chosen JSON output.

    Raises:
        FileNotFoundError: If no JSON output is found.
    """
    for name in ("output.json", "output_file.json"):
        p = job_dir / name
        if p.exists():
            return p
    jsons = sorted(job_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No JSON output found in job dir: {job_dir}")
    return jsons[0]


def _resolve_path(value: Any, *, base_dir: Path) -> str | None:
    """Resolve a path-like config value to an absolute path string.

    Paths beginning with '.testdata/' are resolved relative to the repository root.
    Other relative paths are resolved relative to base_dir.

    Args:
        value: Path-like config value.
        base_dir: Directory to resolve relative paths against.

    Returns:
        Absolute path string or None.
    """
    if value is None:
        return None
    s = str(value)
    if not s:
        return None

    s_norm = s.replace("\\", "/")
    if s_norm.startswith(".testdata/"):
        repo_root = _repo_root_from_this_file()
        return str((repo_root / s_norm).resolve())

    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_path_list(value: Any, *, base_dir: Path) -> list[str]:
    """Resolve a config value representing a path or list of paths.

    Args:
        value: Path-like value or list of path-like values.
        base_dir: Directory to resolve relative paths against.

    Returns:
        List of absolute path strings.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            rp = _resolve_path(v, base_dir=base_dir)
            if rp:
                out.append(rp)
        return out
    rp = _resolve_path(value, base_dir=base_dir)
    return [rp] if rp else []


def _abspathify_config_paths(
    config: dict[str, Any], *, base_dir: Path
) -> dict[str, Any]:
    """Convert configured input paths into absolute paths.

    Args:
        config: Parsed config mapping.
        base_dir: Base directory for resolving relative paths.

    Returns:
        A new config dict with resolved paths.
    """
    path_keys = {"force_file"}
    list_path_keys = {"top_traj_file"}

    out: dict[str, Any] = {}
    for run_name, run_cfg in config.items():
        if not isinstance(run_cfg, dict):
            out[run_name] = run_cfg
            continue

        run_cfg2 = dict(run_cfg)
        for k in list(run_cfg2.keys()):
            if k in path_keys:
                run_cfg2[k] = _resolve_path(run_cfg2.get(k), base_dir=base_dir)
            if k in list_path_keys:
                run_cfg2[k] = _resolve_path_list(run_cfg2.get(k), base_dir=base_dir)

        out[run_name] = run_cfg2

    return out


def _assert_inputs_exist(cooked: dict[str, Any]) -> None:
    """Assert that required input files referenced in cooked config exist."""
    run1 = cooked.get("run1")
    if not isinstance(run1, dict):
        return

    for p in run1.get("top_traj_file") or []:
        if isinstance(p, str) and p:
            assert Path(p).exists(), f"Missing input file: {p}"

    ff = run1.get("force_file")
    if isinstance(ff, str) and ff.strip():
        assert Path(ff).exists(), f"Missing force file: {ff}"


def run_codeentropy_with_config(*, workdir: Path, config_src: Path) -> RunResult:
    """Run CodeEntropy using a regression config file.

    This function loads the YAML config, resolves input paths, ensures required
    dataset files exist by downloading from the filestore if needed, then runs
    CodeEntropy and returns the parsed output JSON.

    Args:
        workdir: Temporary working directory for running CodeEntropy.
        config_src: Path to the YAML regression config.

    Returns:
        RunResult containing outputs and metadata.

    Raises:
        RuntimeError: If CodeEntropy fails or required data cannot be fetched.
        ValueError: If config does not parse as a dict or output JSON lacks expected
        keys.
        FileNotFoundError: If job output files cannot be found.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(config_src.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"Config must parse to a dict. Got {type(raw)} from {config_src}"
        )

    system = config_src.parent.name
    cooked = _abspathify_config_paths(raw, base_dir=config_src.parent)

    required: list[Path] = []
    run1 = cooked.get("run1")
    if isinstance(run1, dict):
        ff = run1.get("force_file")
        if isinstance(ff, str) and ff:
            required.append(Path(ff))
        for p in run1.get("top_traj_file") or []:
            if isinstance(p, str) and p:
                required.append(Path(p))

    if required:
        ensure_testdata_for_system(system, required_paths=required)

    _assert_inputs_exist(cooked)

    (workdir / "config.yaml").write_text(yaml.safe_dump(cooked, sort_keys=False))

    proc = subprocess.run(
        ["python", "-m", "CodeEntropy"],
        cwd=str(workdir),
        capture_output=True,
        text=True,
        env={**os.environ},
    )

    (workdir / "codeentropy_stdout.txt").write_text(proc.stdout or "")
    (workdir / "codeentropy_stderr.txt").write_text(proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(
            "CodeEntropy regression run failed\n"
            f"cwd={workdir}\n"
            f"stdout saved to: {workdir / 'codeentropy_stdout.txt'}\n"
            f"stderr saved to: {workdir / 'codeentropy_stderr.txt'}\n"
        )

    job_dir = _find_latest_job_dir(workdir)
    out_json = _pick_output_json(job_dir)
    payload = json.loads(out_json.read_text())

    (workdir / "codeentropy_output.json").write_text(json.dumps(payload, indent=2))

    if "groups" not in payload:
        raise ValueError(
            f"Regression output JSON did not contain 'groups'. output={out_json}"
        )

    return RunResult(
        workdir=workdir,
        job_dir=job_dir,
        output_json=out_json,
        payload=payload,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
