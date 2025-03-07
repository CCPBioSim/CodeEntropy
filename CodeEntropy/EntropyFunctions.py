import math

# import matplotlib.pyplot as plt
# import MDAnalysis as mda
import numpy as np

# import pandas as pd
from numpy import linalg as la

from CodeEntropy import ConformationFunctions as CONF
from CodeEntropy import UnitsAndConversions as UAC

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
    frequencies = 1 / (2 * pi) * np.sqrt(lambdas / kT)

    return frequencies


# END frequency_calculation


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
    lambdas = UAC.change_lambda_units(lambdas)

    # Calculate frequencies from the eigenvalues
    frequencies = frequency_calculation(lambdas, temp)

    # Sort frequencies lowest to highest
    frequencies = np.sort(frequencies)

    kT = UAC.get_KT2J(temp)
    exponent = UAC.PLANCK_CONST * frequencies / kT
    power_positive = np.power(np.e, exponent)
    power_negative = np.power(np.e, -exponent)
    S_components = exponent / (power_positive - 1) - np.log(1 - power_negative)
    S_components = (
        S_components * UAC.GAS_CONST
    )  # multiply by R - get entropy in J mol^{-1} K^{-1}
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

    return S_vib_total


# END vibrational_entropy


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

    # For each frame, convert the conformation of all dihedrals into a state string
    states = ["" for x in range(number_frames)]
    for frame_index in range(number_frames):
        for index in range(num_dihedrals):
            states[frame_index] += str(conformation[index][frame_index])

    # Count how many times each state occurs, then use the probability to get the
    # entropy
    # entropy = sum over states p*ln(p)
    values, counts = np.unique(states, return_counts=True)
    for state in range(len(values)):
        count = counts[state]
        probability = count / number_frames
        entropy = probability * np.log(probability)
        S_conf_total += entropy

    # multiply by gas constant to get the units J/mol/K
    S_conf_total *= -1 * UAC.GAS_CONST

    return S_conf_total


# END conformational_entropy


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
            # orientational entropy arising from each neighbouring species
            # - we know the species is going to be a neighbour
            S_or_component = math.log(omega)
            S_or_component *= UAC.GAS_CONST
        S_or_total += S_or_component
    # TODO for future releases
    # implement a case for molecules with hydrogen bonds but to a lesser
    # extent than water
    return S_or_total


# END orientational_entropy
