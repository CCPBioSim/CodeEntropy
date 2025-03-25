import argparse
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
        "help": "Stop analysing the trajectory at this frame index",
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
    "outfile": {
        "type": str,
        "help": "Name of the file where the output will be written",
        "default": "outfile.json",
    },
    "mout": {
        "type": str,
        "help": "Name of the file where certain matrices will be written",
        "default": None,
    },
    "force_partitioning": {"type": float, "help": "Force partitioning", "default": 0.5},
    "waterEntropy": {"type": bool, "help": "Calculate water entropy", "default": False},
}


class ConfigManager:
    def __init__(self):
        self.arg_map = arg_map

    def load_config(self, file_path):
        """Load YAML configuration file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

            # If YAML content is empty, return an empty dictionary
            if config is None:
                config = {}

        return config

    def setup_argparse(self):
        """Setup argument parsing dynamically based on arg_map."""
        parser = argparse.ArgumentParser(
            description="CodeEntropy: Entropy calculation with MCC method."
        )

        for arg, properties in self.arg_map.items():
            kwargs = {key: properties[key] for key in properties if key != "help"}
            parser.add_argument(f"--{arg}", **kwargs, help=properties.get("help"))

        return parser

    def merge_configs(self, args, run_config):
        """Merge CLI arguments with YAML configuration and adjust logging level."""
        if run_config is None:
            run_config = {}

        if not isinstance(run_config, dict):
            raise TypeError("run_config must be a dictionary or None.")

        # Step 1: Merge YAML configuration into args
        for key, value in run_config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

        # Step 2: Set default values for any missing arguments from `arg_map`
        for key, params in self.arg_map.items():
            if getattr(args, key, None) is None:
                setattr(args, key, params.get("default"))

        # Step 3: Override with CLI values if provided
        for key in self.arg_map.keys():
            cli_value = getattr(args, key, None)
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
