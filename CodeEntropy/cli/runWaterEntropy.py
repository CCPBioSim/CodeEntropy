#!/usr/bin/env python

"""
CLI script for reading a topology and coordinate trajectory to get 
the orientational entropy of interfacial water molecules
"""

import argparse
import logging
import sys
from datetime import datetime
from MDAnalysis import Universe

import CodeEntropy.neighbours.interfacial_solvent as GetSolvent


def run_force_pred(
    file_topology="file_topology",
    file_coords="file_coords",
    file_forces="file_forces",
    start="start",
    end="end",
    step="step",
):
    # pylint: disable=unused-argument
    """
    Read a topology and coordinate trajectory to get the orientational
    entropy of interfacial water molecules
    """

    startTime = datetime.now()
    print(startTime)

    system = Universe(file_topology, file_coords)
    print(system.trajectory)

    GetSolvent.get_interfacial_water_orient_entropy(system, start-1, 
                                                    end-1, step)

    sys.stdout.flush()
    print("end")
    print(datetime.now() - startTime)


def main():
    """ """
    try:
        usage = "runWaterEntropy.py [-h]"
        parser = argparse.ArgumentParser(
            description="Program for reading "
            "in molecule forces, coordinates and energies for "
            "entropy calculations.",
            usage=usage,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument_group("Options")
        parser.add_argument(
            "-t",
            "--file_topology",
            metavar="file",
            default=None,
            help="name of file containing system topology.",
        )
        parser.add_argument(
            "-c",
            "--file_coords",
            metavar="file",
            default=None,
            help="name of file containing coordinates.",
        )
        parser.add_argument(
            "-f",
            "--file_forces",
            metavar="file",
            default=None,
            help="name of file containing forces.",
        )
        parser.add_argument('-s', '--start', action='store', type=int,
                        default='1', help='starting frame number')
        parser.add_argument('-e', '--end', action='store', type=int,
                default='1', help='end frame number')
        parser.add_argument('-dt', '--step', action='store', type=int,
                default='1', help='steps between frames')
        op = parser.parse_args()
    except argparse.ArgumentError:
        logging.error(
            "Command line arguments are ill-defined, please check the arguments."
        )
        raise
        sys.exit(1)

    run_force_pred(
        file_topology=op.file_topology,
        file_coords=op.file_coords,
        file_forces=op.file_forces,
        start=op.start,
        end=op.end,
        step=op.step,
    )


if __name__ == "__main__":
    main()
