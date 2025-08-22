import argparse
import glob
import logging
import os

import yaml

# Set up logger
logger = logging.getLogger(__name__)

arg_map = {
    "top_traj_file": {
        "type": str,
        "nargs": "+",
        "help": "Path to Structure/topology file followed by Trajectory file(s)",
    },
    "selection_string": {
        "type": str,
        "help": "Selection string for CodeEntropy",
        "default": "all",
    },
    "start": {
        "type": int,
        "help": "Start analysing the trajectory from this frame index",
        "default": 0,
    },
    "end": {
        "type": int,
        "help": (
            "Stop analysing the trajectory at this frame index. This is "
            "the frame index of the last frame to be included, so for example"
            "if start=0 and end=500 there would be 501 frames analysed. The "
            "default -1 will include the last frame."
        ),
        "default": -1,
    },
    "step": {
        "type": int,
        "help": "Interval between two consecutive frames to be read index",
        "default": 1,
    },
    "bin_width": {
        "type": int,
        "help": "Bin width in degrees for making the histogram",
        "default": 30,
    },
    "temperature": {
        "type": float,
        "help": "Temperature for entropy calculation (K)",
        "default": 298.0,
    },
    "verbose": {
        "action": "store_true",
        "help": "Enable verbose output",
    },
    "thread": {"type": int, "help": "How many multiprocess to use", "default": 1},
    "output_file": {
        "type": str,
        "help": "Name of the file where the output will be written",
        "default": "output_file.json",
    },
    "force_partitioning": {"type": float, "help": "Force partitioning", "default": 0.5},
    "water_entropy": {
        "type": bool,
        "help": "If set to False, disables the calculation of water entropy",
        "default": True,
    },
    "grouping": {
        "type": str,
        "help": "How to group molecules for averaging",
        "default": "molecules",
    },
}


class ConfigManager:
    def __init__(self):
        self.arg_map = arg_map

    def load_config(self, file_path):
        """Load YAML configuration file from the given directory."""
        yaml_files = glob.glob(os.path.join(file_path, "*.yaml"))

        if not yaml_files:
            return {"run1": {}}

        try:
            with open(yaml_files[0], "r") as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from: {yaml_files[0]}")
                if config is None:
                    config = {"run1": {}}
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            config = {"run1": {}}

        return config

    def str2bool(self, value):
        """
        Convert a string or boolean input into a boolean value.

        Accepts common string representations of boolean values such as:
        - True values: "true", "t", "yes", "1"
        - False values: "false", "f", "no", "0"

        If the input is already a boolean, it is returned as-is.
        Raises:
            argparse.ArgumentTypeError: If the input cannot be interpreted as a boolean.

        Args:
            value (str or bool): The input value to convert.

        Returns:
            bool: The corresponding boolean value.
        """
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in {"true", "t", "yes", "1"}:
            return True
        elif value in {"false", "f", "no", "0"}:
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

    def setup_argparse(self):
        """Setup argument parsing dynamically based on arg_map."""
        parser = argparse.ArgumentParser(
            description="CodeEntropy: Entropy calculation with MCC method."
        )

        for arg, properties in self.arg_map.items():
            help_text = properties.get("help", "")
            default = properties.get("default", None)

            if properties.get("type") == bool:
                parser.add_argument(
                    f"--{arg}",
                    type=self.str2bool,
                    default=default,
                    help=f"{help_text} (default: {default})",
                )
            else:
                kwargs = {k: v for k, v in properties.items() if k != "help"}
                parser.add_argument(f"--{arg}", **kwargs, help=help_text)

        return parser

    def merge_configs(self, args, run_config):
        """Merge CLI arguments with YAML configuration and adjust logging level."""
        if run_config is None:
            run_config = {}

        if not isinstance(run_config, dict):
            raise TypeError("run_config must be a dictionary or None.")

        # Convert argparse Namespace to dictionary
        args_dict = vars(args)

        # Reconstruct parser and check which arguments were explicitly provided via CLI
        parser = self.setup_argparse()
        default_args = parser.parse_args([])
        default_dict = vars(default_args)

        cli_provided_args = {
            key for key, value in args_dict.items() if value != default_dict.get(key)
        }

        # Step 1: Apply YAML values if CLI didn't explicitly set the argument
        for key, yaml_value in run_config.items():
            if yaml_value is not None and key not in cli_provided_args:
                logger.debug(f"Using YAML value for {key}: {yaml_value}")
                setattr(args, key, yaml_value)

        # Step 2: Ensure all arguments have at least their default values
        for key, params in self.arg_map.items():
            if getattr(args, key, None) is None:
                setattr(args, key, params.get("default"))

        # Step 3: Ensure CLI arguments always take precedence
        for key in self.arg_map.keys():
            cli_value = args_dict.get(key)
            if cli_value is not None:
                run_config[key] = cli_value

        # Adjust logging level based on 'verbose' flag
        if getattr(args, "verbose", False):
            logger.setLevel(logging.DEBUG)
            for handler in logger.handlers:
                handler.setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled. Logger set to DEBUG level.")
        else:
            logger.setLevel(logging.INFO)
            for handler in logger.handlers:
                handler.setLevel(logging.INFO)

        return args

    def input_parameters_validation(self, u, args):
        """Check the validity of the user inputs against sensible values"""

        self._check_input_start(u, args)
        self._check_input_end(u, args)
        self._check_input_step(args)
        self._check_input_bin_width(args)
        self._check_input_temperature(args)
        self._check_input_force_partitioning(args)

    def _check_input_start(self, u, args):
        """Check that the input does not exceed the length of the trajectory."""
        if args.start > len(u.trajectory):
            raise ValueError(
                f"Invalid 'start' value: {args.start}. It exceeds the trajectory length"
                " of {len(u.trajectory)}."
            )

    def _check_input_end(self, u, args):
        """Check that the end index does not exceed the trajectory length."""
        if args.end > len(u.trajectory):
            raise ValueError(
                f"Invalid 'end' value: {args.end}. It exceeds the trajectory length of"
                " {len(u.trajectory)}."
            )

    def _check_input_step(self, args):
        """Check that the step value is non-negative."""
        if args.step < 0:
            logger.warning(
                f"Negative 'step' value provided: {args.step}. This may lead to"
                " unexpected behavior."
            )

    def _check_input_bin_width(self, args):
        """Check that the bin width is within the valid range [0, 360]."""
        if args.bin_width < 0 or args.bin_width > 360:
            raise ValueError(
                f"Invalid 'bin_width': {args.bin_width}. It must be between 0 and 360"
                " degrees."
            )

    def _check_input_temperature(self, args):
        """Check that the temperature is non-negative."""
        if args.temperature < 0:
            raise ValueError(
                f"Invalid 'temperature': {args.temperature}. Temperature cannot be"
                " below 0."
            )

    def _check_input_force_partitioning(self, args):
        """Warn if force partitioning is not set to the default value."""
        default_value = arg_map["force_partitioning"]["default"]
        if args.force_partitioning != default_value:
            logger.warning(
                f"'force_partitioning' is set to {args.force_partitioning},"
                " which differs from the default ({default_value})."
            )
