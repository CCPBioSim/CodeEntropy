"""Configuration and CLI argument management for CodeEntropy.

This module provides:

1) A declarative argument specification (`ARG_SPECS`) used to build an
   ``argparse.ArgumentParser``.
2) A `ConfigResolver` that:
   - loads YAML configuration (if present),
   - merges YAML values with CLI values (CLI wins),
   - adjusts logging verbosity,
   - validates a subset of runtime inputs against the trajectory.

Notes:
- Boolean arguments are parsed via `str2bool` to support YAML/CLI interop and
  common string forms like "true"/"false".
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgSpec:
    """Argument specification used to build an argparse parser.

    Attributes:
        help: Help text shown in CLI usage.
        default: Default value if not provided via CLI or YAML.
        type: Python type for parsing (e.g., int, float, str, bool). If bool,
            `ConfigResolver.str2bool` will be used.
        action: Optional argparse action (e.g., "store_true").
        nargs: Optional nargs spec (e.g., "+").
    """

    help: str
    default: Any = None
    type: Any = None
    action: Optional[str] = None
    nargs: Optional[str] = None


ARG_SPECS: Dict[str, ArgSpec] = {
    "top_traj_file": ArgSpec(
        type=str,
        nargs="+",
        help="Path to structure/topology file followed by trajectory file",
    ),
    "force_file": ArgSpec(
        type=str,
        default=None,
        help="Optional path to force file if forces are not in trajectory file",
    ),
    "file_format": ArgSpec(
        type=str,
        default=None,
        help="String for file format as recognised by MDAnalysis",
    ),
    "kcal_force_units": ArgSpec(
        type=bool,
        default=False,
        help="Set this to True if you have a separate force file with kcal units.",
    ),
    "selection_string": ArgSpec(
        type=str,
        default="all",
        help="Selection string for CodeEntropy",
    ),
    "start": ArgSpec(
        type=int,
        default=0,
        help="Start analysing the trajectory from this frame index",
    ),
    "end": ArgSpec(
        type=int,
        default=-1,
        help=(
            "Stop analysing the trajectory at this frame index. This is "
            "the frame index of the last frame to be included, so for example "
            "if start=0 and end=500 there would be 501 frames analysed. The "
            "default -1 will include the last frame."
        ),
    ),
    "step": ArgSpec(
        type=int,
        default=1,
        help="Interval between two consecutive frames to be read index",
    ),
    "bin_width": ArgSpec(
        type=int,
        default=30,
        help="Bin width in degrees for making the histogram",
    ),
    "temperature": ArgSpec(
        type=float,
        default=298.0,
        help="Temperature for entropy calculation (K)",
    ),
    "verbose": ArgSpec(
        action="store_true",
        help="Enable verbose output",
    ),
    "output_file": ArgSpec(
        type=str,
        default="output_file.json",
        help=(
            "Name of the output file to write results to (filename only). Defaults "
            "to output_file.json"
        ),
    ),
    "force_partitioning": ArgSpec(
        type=float,
        default=0.5,
        help="Force partitioning",
    ),
    "water_entropy": ArgSpec(
        type=bool,
        default=True,
        help="If set to False, disables the calculation of water entropy",
    ),
    "grouping": ArgSpec(
        type=str,
        default="molecules",
        help="How to group molecules for averaging",
    ),
    "combined_forcetorque": ArgSpec(
        type=bool,
        default=True,
        help="Use combined force-torque matrix for residue level vibrational entropies",
    ),
    "customised_axes": ArgSpec(
        type=bool,
        default=True,
        help="Use bonded axes to rotate forces for UA level vibrational entropies",
    ),
    "search_type": ArgSpec(
        type=str,
        default="RAD",
        help="Type of neighbor search to use."
        "Default RAD; grid search is also available",
    ),
}


class ConfigResolver:
    """Load, merge, and validate CodeEntropy configuration.

    This class provides a consistent interface for:
      - YAML config discovery/loading
      - CLI parser construction
      - merging YAML values with CLI values (CLI wins)
      - setting logging verbosity
      - validating trajectory-related numeric parameters
    """

    def __init__(self, arg_specs: Optional[Dict[str, ArgSpec]] = None) -> None:
        """Initialize the manager.

        Args:
            arg_specs: Optional override for argument specs. If omitted, uses
                `ARG_SPECS`.
        """
        self._arg_specs = dict(arg_specs or ARG_SPECS)

    def load_config(self, directory_path: str) -> Dict[str, Any]:
        """Load the first YAML config file found in a directory.

        The current behavior matches your existing workflow:
        - searches for ``*.yaml`` in `directory_path`,
        - loads the first match,
        - returns ``{"run1": {}}`` if none found or file is empty/invalid.

        Args:
            directory_path: Directory to search for YAML files.

        Returns:
            A configuration dictionary.
        """
        yaml_files = glob.glob(os.path.join(directory_path, "*.yaml"))
        if not yaml_files:
            return {"run1": {}}

        config_path = yaml_files[0]
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {"run1": {}}
            logger.info("Loaded configuration from: %s", config_path)
            return config
        except Exception as exc:
            logger.error("Failed to load config file: %s", exc)
            return {"run1": {}}

    @staticmethod
    def str2bool(value: Any) -> bool:
        """Convert a string or boolean input into a boolean.

        Accepts common string representations:
          - True values: "true", "t", "yes", "1"
          - False values: "false", "f", "no", "0"

        If the input is already a boolean, it is returned as-is.

        Args:
            value: Input value to convert.

        Returns:
            The corresponding boolean.

        Raises:
            argparse.ArgumentTypeError: If the input cannot be interpreted as a boolean.
        """
        if isinstance(value, bool):
            return value
        if not isinstance(value, str):
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

        lowered = value.lower()
        if lowered in {"true", "t", "yes", "1"}:
            return True
        if lowered in {"false", "f", "no", "0"}:
            return False
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    def build_parser(self) -> argparse.ArgumentParser:
        """Build an ArgumentParser from argument specs.

        Returns:
            An argparse.ArgumentParser configured with all supported flags.
        """
        parser = argparse.ArgumentParser(
            description="CodeEntropy: Entropy calculation with MCC method."
        )

        for name, spec in self._arg_specs.items():
            arg_name = f"--{name}"

            if spec.action is not None:
                parser.add_argument(arg_name, action=spec.action, help=spec.help)
                continue

            if spec.type is bool:
                parser.add_argument(
                    arg_name,
                    type=self.str2bool,
                    default=spec.default,
                    help=f"{spec.help} (default: {spec.default})",
                )
                continue

            kwargs: Dict[str, Any] = {}
            if spec.type is not None:
                kwargs["type"] = spec.type
            if spec.default is not None:
                kwargs["default"] = spec.default
            if spec.nargs is not None:
                kwargs["nargs"] = spec.nargs

            parser.add_argument(arg_name, help=spec.help, **kwargs)

        return parser

    def resolve(
        self, args: argparse.Namespace, run_config: Optional[Dict[str, Any]]
    ) -> argparse.Namespace:
        """Merge CLI arguments with YAML configuration and adjust logging level.

        Merge rule:
          - CLI explicitly-provided values take precedence.
          - YAML values fill in missing values.
          - Defaults fill in anything still unset.

        Args:
            args: Parsed CLI arguments.
            run_config: Dict of YAML values for a specific run, or None.

        Returns:
            The mutated argparse.Namespace with merged values.

        Raises:
            TypeError: If `run_config` is not a dict or None.
        """
        if run_config is None:
            run_config = {}
        if not isinstance(run_config, dict):
            raise TypeError("run_config must be a dictionary or None.")

        args_dict = vars(args)

        parser = self.build_parser()
        default_args = parser.parse_args([])
        default_dict = vars(default_args)

        cli_provided = self._detect_cli_overrides(args_dict, default_dict)

        self._apply_yaml_defaults(args, run_config, cli_provided)
        self._ensure_defaults(args)
        self._apply_logging_level(bool(getattr(args, "verbose", False)))

        return args

    @staticmethod
    def _detect_cli_overrides(
        args_dict: Dict[str, Any], default_dict: Dict[str, Any]
    ) -> Set[str]:
        """Detect which args were explicitly overridden in the CLI.

        Args:
            args_dict: Parsed arg values.
            default_dict: Parser defaults.

        Returns:
            Set of argument names that differ from defaults.
        """
        return {k for k, v in args_dict.items() if v != default_dict.get(k)}

    def _apply_yaml_defaults(
        self,
        args: argparse.Namespace,
        run_config: Dict[str, Any],
        cli_provided: Set[str],
    ) -> None:
        """Apply YAML values onto args for keys not provided by CLI.

        Args:
            args: Parsed CLI arguments (mutated in-place).
            run_config: YAML dict for this run.
            cli_provided: Keys explicitly set via CLI.
        """
        for key, yaml_value in run_config.items():
            if yaml_value is None or key in cli_provided:
                continue
            if key in self._arg_specs:
                logger.debug("Using YAML value for %s: %s", key, yaml_value)
                setattr(args, key, yaml_value)

    def _ensure_defaults(self, args: argparse.Namespace) -> None:
        """Ensure all known args have defaults if still unset.

        Args:
            args: Parsed arg namespace (mutated in-place).
        """
        for key, spec in self._arg_specs.items():
            if getattr(args, key, None) is None:
                setattr(args, key, spec.default)

    @staticmethod
    def _apply_logging_level(verbose: bool) -> None:
        """Adjust logging levels for this module's logger and its handlers.

        Args:
            verbose: Whether to enable DEBUG logging.
        """
        level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        if verbose:
            logger.debug("Verbose mode enabled. Logger set to DEBUG level.")

    def validate_inputs(self, u: Any, args: argparse.Namespace) -> None:
        """Validate user inputs against sensible runtime constraints.

        Args:
            u: MDAnalysis universe (or compatible) with a `trajectory`.
            args: Parsed/merged arguments.

        Raises:
            ValueError: If a parameter is invalid.
        """
        self._check_input_start(u, args)
        self._check_input_end(u, args)
        self._check_input_step(args)
        self._check_input_bin_width(args)
        self._check_input_temperature(args)
        self._check_input_force_partitioning(args)

    @staticmethod
    def _check_input_start(u: Any, args: argparse.Namespace) -> None:
        """Check that the start index does not exceed the trajectory length."""
        traj_len = len(u.trajectory)
        if args.start > traj_len:
            raise ValueError(
                f"Invalid 'start' value: {args.start}. It exceeds the trajectory "
                f"length of {traj_len}."
            )

    @staticmethod
    def _check_input_end(u: Any, args: argparse.Namespace) -> None:
        """Check that the end index does not exceed the trajectory length."""
        traj_len = len(u.trajectory)
        if args.end > traj_len:
            raise ValueError(
                f"Invalid 'end' value: {args.end}. It exceeds the trajectory length of "
                f"{traj_len}."
            )

    @staticmethod
    def _check_input_step(args: argparse.Namespace) -> None:
        """Warn if the step value is negative."""
        if args.step < 0:
            logger.warning(
                "Negative 'step' value provided: %s. This may lead to unexpected "
                "behavior.",
                args.step,
            )

    @staticmethod
    def _check_input_bin_width(args: argparse.Namespace) -> None:
        """Check that the bin width is within the valid range [0, 360]."""
        if args.bin_width < 0 or args.bin_width > 360:
            raise ValueError(
                f"Invalid 'bin_width': {args.bin_width}. It must be between 0 and 360 "
                f"degrees."
            )

    @staticmethod
    def _check_input_temperature(args: argparse.Namespace) -> None:
        """Check that the temperature is non-negative."""
        if args.temperature < 0:
            raise ValueError(
                f"Invalid 'temperature': {args.temperature}. Temperature cannot be "
                f"below 0."
            )

    def _check_input_force_partitioning(self, args: argparse.Namespace) -> None:
        """Warn if force partitioning is not set to the default value."""
        default_value = self._arg_specs["force_partitioning"].default
        if args.force_partitioning != default_value:
            logger.warning(
                "'force_partitioning' is set to %s, which differs from the default %s.",
                args.force_partitioning,
                default_value,
            )
