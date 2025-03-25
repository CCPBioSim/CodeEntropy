import logging
import math

# import matplotlib.pyplot as plt
# import MDAnalysis as mda
import numpy as np

# import pandas as pd
from numpy import linalg as la

from CodeEntropy.calculations import ConformationFunctions as CONF
from CodeEntropy.calculations import UnitsAndConversions as UAC

logger = logging.getLogger(__name__)

# import sys
# from ast import arg


# from CodeEntropy import NeighbourFunctions as NF


def frequency_calculation(lambdas, temp):
    """
    Function to calculate an array of vibrational frequencies from the eigenvalues of
    the covariance matrix.

    Calculated from eq. (3) in Higham, S.-Y. Chou, F. Gräter and  R. H. Henchman,
    Molecular Physics, 2018, 116, 1965–1976//eq. (3) in A. Chakravorty, J. Higham
    and R. H. Henchman, J. Chem. Inf. Model., 2020, 60, 5540–5551

    frequency=sqrt(λ/kT)/2π

    Input
    -----
       lambdas : array of floats - eigenvalues of the covariance matrix
       temp: float - temperature

    Returns
    -------
       frequencies : array of floats - corresponding vibrational frequencies
    """
    pi = np.pi
    # get kT in Joules from given temperature
    kT = UAC.get_KT2J(temp)
    logger.debug(f"Temperature: {temp}, kT: {kT}")

    lambdas = np.array(lambdas)  # Ensure input is a NumPy array
    logger.debug(f"Eigenvalues (lambdas): {lambdas}")

    # Check for negatives and raise an error if any are found
    if np.any(lambdas < 0):
        logger.error(f"Negative eigenvalues encountered: {lambdas[lambdas < 0]}")
        raise ValueError(f"Negative eigenvalues encountered: {lambdas[lambdas < 0]}")

    # Compute frequencies safely
    frequencies = 1 / (2 * pi) * np.sqrt(lambdas / kT)
    logger.debug(f"Calculated frequencies: {frequencies}")

    return frequencies


def vibrational_entropy(matrix, matrix_type, temp, highest_level):
    """
    Function to calculate the vibrational entropy for each level calculated from eq. (4)
    in J. Higham, S.-Y. Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018,
    116, 1965–1976 / eq. (2) in A. Chakravorty, J. Higham and R. H. Henchman,
    J. Chem. Inf. Model., 2020, 60, 5540–5551.

    Input
    -----
       matrix : matrix - force/torque covariance matrix
       matrix_type: string
       temp: float - temperature
       highest_level: bool - is this the highest level of the heirarchy (whole molecule)

    Returns
    -------
       S_vib_total : float - transvibrational/rovibrational entropy
    """
    # N beads at a level => 3N x 3N covariance matrix => 3N eigenvalues
    # Get eigenvalues of the given matrix and change units to SI units
    lambdas = la.eigvals(matrix)
    logger.debug(f"Eigenvalues (lambdas) before unit change: {lambdas}")
    lambdas = UAC.change_lambda_units(lambdas)
    logger.debug(f"Eigenvalues (lambdas) after unit change: {lambdas}")

    # Calculate frequencies from the eigenvalues
    frequencies = frequency_calculation(lambdas, temp)
    logger.debug(f"Calculated frequencies: {frequencies}")

    # Sort frequencies lowest to highest
    frequencies = np.sort(frequencies)
    logger.debug(f"Sorted frequencies: {frequencies}")

    kT = UAC.get_KT2J(temp)
    logger.debug(f"Temperature: {temp}, kT: {kT}")
    exponent = UAC.PLANCK_CONST * frequencies / kT
    logger.debug(f"Exponent values: {exponent}")
    power_positive = np.power(np.e, exponent)
    power_negative = np.power(np.e, -exponent)
    logger.debug(f"Power positive values: {power_positive}")
    logger.debug(f"Power negative values: {power_negative}")
    S_components = exponent / (power_positive - 1) - np.log(1 - power_negative)
    S_components = (
        S_components * UAC.GAS_CONST
    )  # multiply by R - get entropy in J mol^{-1} K^{-1}
    logger.debug(f"Entropy components: {S_components}")
    # N beads at a level => 3N x 3N covariance matrix => 3N eigenvalues
    if matrix_type == "force":  # force covariance matrix
        if highest_level:  # whole molecule level - we take all frequencies into account
            S_vib_total = sum(S_components)

        # discard the 6 lowest frequencies to discard translation and rotation of the
        # whole unit the overall translation and rotation of a unit is an internal
        # motion of the level above
        else:
            S_vib_total = sum(S_components[6:])

    else:  # torque covariance matrix - we always take all values into account
        S_vib_total = sum(S_components)

    logger.debug(f"Total vibrational entropy: {S_vib_total}")

    return S_vib_total


def conformational_entropy(
    data_container, dihedrals, bin_width, start, end, step, number_frames
):
    """
    Function to calculate conformational entropies using eq. (7) in Higham, S.-Y. Chou,
    F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116, 1965–1976 / eq. (4) in
    A. Chakravorty, J. Higham and R. H. Henchman, J. Chem. Inf. Model., 2020, 60,
    5540–5551.

    Uses the adaptive enumeration method (AEM).

    Input
    -----
    dihedrals : array - array of dihedrals in the molecule
    Returns
    -------
       S_conf_total : float - conformational entropy
    """

    S_conf_total = 0

    # For each dihedral, identify the conformation in each frame
    num_dihedrals = len(dihedrals)
    conformation = np.zeros((num_dihedrals, number_frames))
    index = 0
    for dihedral in dihedrals:
        conformation[index] = CONF.assign_conformation(
            data_container, dihedral, number_frames, bin_width, start, end, step
        )
        index += 1

    logger.debug(f"Conformation matrix: {conformation}")

    # For each frame, convert the conformation of all dihedrals into a state string
    states = ["" for x in range(number_frames)]
    for frame_index in range(number_frames):
        for index in range(num_dihedrals):
            states[frame_index] += str(conformation[index][frame_index])

    logger.debug(f"States: {states}")

    # Count how many times each state occurs, then use the probability to get the
    # entropy
    # entropy = sum over states p*ln(p)
    values, counts = np.unique(states, return_counts=True)
    for state in range(len(values)):
        logger.debug(f"Unique states: {values}")
        logger.debug(f"Counts: {counts}")
        count = counts[state]
        probability = count / number_frames
        entropy = probability * np.log(probability)
        S_conf_total += entropy

    # multiply by gas constant to get the units J/mol/K
    S_conf_total *= -1 * UAC.GAS_CONST

    logger.debug(f"Total conformational entropy: {S_conf_total}")

    return S_conf_total


def orientational_entropy(neighbours_dict):
    """
    Function to calculate orientational entropies from eq. (10) in J. Higham, S.-Y.
    Chou, F. Gräter and R. H. Henchman, Molecular Physics, 2018, 116,3 1965–1976.
    Number of orientations, Ω, is calculated using eq. (8) in  J. Higham,
    S.-Y. Chou, F. Gräter and R. H. Henchman,  Molecular Physics, 2018, 116,
    3 1965–1976.

    σ is assumed to be 1 for the molecules we're concerned with and hence,
    max {1, (Nc^3*π)^(1/2)} will always be (Nc^3*π)^(1/2).

    TODO future release - function for determing symmetry and symmetry numbers
    maybe?

    Input
    -----
      neighbours_dict :  dictionary - dictionary of neighbours for the molecule -
          should contain the type of neighbour molecule and the number of neighbour
          molecules of that species

    Returns
    -------
       S_or_total : float - orientational entropy
    """

    # Replaced molecule with neighbour as this is what the for loop uses
    S_or_total = 0
    for neighbour in neighbours_dict:  # we are going through neighbours
        if neighbour in []:  # water molecules - call POSEIDON functions
            pass  # TODO temporary until function is written
        else:
            # the bound ligand is always going to be a neighbour
            omega = np.sqrt((neighbours_dict[neighbour] ** 3) * math.pi)
            logger.debug(f"Omega for neighbour {neighbour}: {omega}")
            # orientational entropy arising from each neighbouring species
            # - we know the species is going to be a neighbour
            S_or_component = math.log(omega)
            logger.debug(
                f"S_or_component (log(omega)) for neighbour {neighbour}: "
                f"{S_or_component}"
            )
            S_or_component *= UAC.GAS_CONST
            logger.debug(
                f"S_or_component after multiplying by GAS_CONST for neighbour "
                f"{neighbour}: {S_or_component}"
            )
        S_or_total += S_or_component
        logger.debug(
            f"S_or_total after adding component for neighbour {neighbour}: "
            f"{S_or_total}"
        )
    # TODO for future releases
    # implement a case for molecules with hydrogen bonds but to a lesser
    # extent than water

    logger.debug(f"Final total orientational entropy: {S_or_total}")

    return S_or_total
