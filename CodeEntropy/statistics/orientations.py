#!/usr/bin/env python

"""
These functions calculate orientational entropy from labelled
coordination shells
"""

from collections import Counter

import numpy as np

from CodeEntropy.utils.helpers import nested_dict


class Labels:
    """
    Labelled shell counts used for Sorient and Smix.
    The counts are placed into two dictionaries used for statistics later,
    the structure of these two dictionaries are as follows:

    dict1 = {"resname": {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}
    dict2 = {"nearest_resid": {"resname":
                                {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}
    """

    labelled_shell_counts = nested_dict()  # save shell instances in here
    resid_labelled_shell_counts = nested_dict()  # save shell instances in here

    def __init__(self, resid, resname, labelled_shell, donates_to, accepts_from):
        # pylint: disable=too-many-arguments
        Labels.add_shell_counts(self, resid, resname, labelled_shell)
        Labels.add_donates_to(self, resid, resname, labelled_shell, donates_to)
        Labels.add_accepts_from(self, resid, resname, labelled_shell, accepts_from)

    def add_shell_counts(self, resid, resname, labelled_shell):
        """
        Add a labelled shell to a dictionary that keeps track of the counts
        for each labelled shell type with constituents alpha-numerically
        ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        """
        labelled_shell = tuple(sorted(labelled_shell))
        if "shell_count" not in Labels.labelled_shell_counts[resname][labelled_shell]:
            Labels.labelled_shell_counts[resname][labelled_shell]["shell_count"] = 1
        else:
            Labels.labelled_shell_counts[resname][labelled_shell]["shell_count"] += 1

        if (
            "shell_count"
            not in Labels.resid_labelled_shell_counts[resid][resname][labelled_shell]
        ):
            Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                "shell_count"
            ] = 1
        else:
            Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                "shell_count"
            ] += 1

    def add_donates_to(self, resid, resname, labelled_shell, donates_to):
        """
        Add a labelled neighbours donated to in a dictionary that keeps
        track of the counts for each labelled shell type with
        constituents alpha-numerically ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        :param donates_to: list of labelled neighbours that are donated to
        """
        labelled_shell = tuple(sorted(labelled_shell))
        # donates_to = tuple(sorted(donates_to))
        for a in donates_to:
            if (
                a
                not in Labels.labelled_shell_counts[resname][labelled_shell][
                    "donates_to"
                ]
            ):
                Labels.labelled_shell_counts[resname][labelled_shell]["donates_to"][
                    a
                ] = 1
            else:
                Labels.labelled_shell_counts[resname][labelled_shell]["donates_to"][
                    a
                ] += 1

            if (
                a
                not in Labels.resid_labelled_shell_counts[resid][resname][
                    labelled_shell
                ]["donates_to"]
            ):
                Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "donates_to"
                ][a] = 1
            else:
                Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "donates_to"
                ][a] += 1

    def add_accepts_from(self, resid, resname, labelled_shell, accepts_from):
        """
        Add a labelled neighbours accepted from to in a dictionary that keeps
        track of the counts for each labelled shell type with
        constituents alpha-numerically ordered

        :param self: class instance
        :param resid: residue id of nearest nonlike atom in the labelled shell
        :param resname: residue name of nearest nonlike atom in the labelled shell
        :param labelled_shell: coordination shell with labelled neighbours
        :param accepts_from: list of labelled neighbours that are accepted_from
        """
        labelled_shell = tuple(sorted(labelled_shell))
        # accepts_from = tuple(sorted(accepts_from))
        for d in accepts_from:
            if (
                d
                not in Labels.labelled_shell_counts[resname][labelled_shell][
                    "accepts_from"
                ]
            ):
                Labels.labelled_shell_counts[resname][labelled_shell]["accepts_from"][
                    d
                ] = 1
            else:
                Labels.labelled_shell_counts[resname][labelled_shell]["accepts_from"][
                    d
                ] += 1

            if (
                d
                not in Labels.resid_labelled_shell_counts[resid][resname][
                    labelled_shell
                ]["accepts_from"]
            ):
                Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "accepts_from"
                ][d] = 1
            else:
                Labels.resid_labelled_shell_counts[resid][resname][labelled_shell][
                    "accepts_from"
                ][d] += 1


def get_running_average(
    value: float, count: int, running_average_value: float, count_stored: int
):
    """
    For a given value, get it's running average from the current value

    :param value: the value that needs to be added to the running average
    :param count: the number of times the value occurs from statistics
    :param running_average_value: the currently stored running average
    :param count_stored: the currently stored count for the running average
    """
    new_count_stored = count_stored + count
    new_running_average = (
        value * count + running_average_value * count_stored
    ) / new_count_stored

    return new_running_average, new_count_stored


def get_resid_orientational_entropy_from_dict(resid_labelled_dict: dict):
    """
    For a given dictionary containing labelled shells and HBing within the
    shell with format:

    RADShell.Labels.resid_labelled_shell_counts
    dict2 = {"nearest_resid": {"resname":
                                {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}

    Get the orientational entropy of the molecules in this dict

    :param resid_labelled_dict: dictionary of format dict2 containing labelled
        coordination shells and HB donating and accepting
    """
    Sorient_dict = nested_dict()
    for resid, shell_label_key in sorted(list(resid_labelled_dict.items())):
        # print("resid", resid)
        Sorient_dict[resid] = get_orientational_entropy_from_dict(shell_label_key)
    return Sorient_dict


def get_orientational_entropy_from_dict(labelled_dict: dict):
    """
    For a given dictionary containing labelled shells and HBing within the
    shell with format:

    RADShell.Labels.labelled_shell_counts

    dict1 = {"resname": {("labelled_shell"): {"shell_count": 0,
                                "donates_to": {"labelled_donators": 0,},
                                "accepts_from": {"labelled_acceptors": 0,}
                                }}}

    Get the orientational entropy of the molecules in this dict

    :param labelled_dict: dictionary of format dict1 containing labelled
        coordination shells and HB donating and accepting
    """
    Sorient_dict = nested_dict()
    for resname, shell_label_key in sorted(list(labelled_dict.items())):
        # print("resname", resname)
        Sorient_ave, tot_count = 0, 0
        for shell_label, vals1 in sorted(list(shell_label_key.items())):
            # print(shell_label, len(shell_label))
            # print("shell_counts", vals1["shell_count"])
            degeneracy = get_shell_degeneracy(shell_label)
            pD_dict = get_donor_acceptor_probabilities(vals1, "donates_to")
            pA_dict = get_donor_acceptor_probabilities(vals1, "accepts_from")
            Nc_eff, pbias_ave = get_reduced_neighbours_biases(
                degeneracy, pD_dict, pA_dict
            )
            # print('pbias_ave', round(pbias_ave, 5))
            # print('Nc_eff', round(Nc_eff, 5))
            S_orient = get_orientation_S(Nc_eff, pbias_ave)
            # print('S_orient', round(S_orient, 5))
            Sorient_ave, tot_count = get_running_average(
                S_orient, vals1["shell_count"], Sorient_ave, tot_count
            )
            # print(">>>", Sorient_ave, tot_count)
        Sorient_dict[resname] = [Sorient_ave, tot_count]
    return Sorient_dict


def get_orientation_S(Nc_eff: float, pbias_ave: float):
    """
    Get the orientational entropy of water molecules, or any single UA molecule
    containing two hydrogen bond donors.

    S_orient = np.log((Nc_eff) ** (3 / 2) * np.pi ** 0.5 * pbias_ave / 2)

    This equation is modified from the previous theory of water molecule
    orientational entropy, where hydrogen bonding is accounted for by reducing
    the number of available neighbours.
    Here, the coordination shell neighbours available to hydrogen bond with
    are reduced to $Nc_eff$. The reduction in available HBing neighbours is
    calculated from statistics gathered from simulation trajectories, where
    neighbour types that are donated to or accepted from are counted.

    Nc_eff = sum_i((ppD_i * ppA_i) * N_i / 0.25)

    where $ppD_i$ is the probability to donate to a given neighbour type i
    compared to accepting from the same neighbour type. $ppA_i$ is the
    equivalent probability for accepting from neighbour type i.

    $pbias_ave$ is the average bias in accepting from and donating to
    a given neighbour type i. If it is equally likely to donate to and
    accept from all neighbour types, then $pbias_ave=0.25$, but if there is
    any bias in preferentially donating to, accepting from or not HBing to a
    neighbour at all, then $pbias_ave < 0.25$.

    $pbias_ave$ = sum_i(ppD_i * ppA_i) / Nc

    Both $Nc_eff$ and $pbias_ave$ are used to reduce the orientational entropy
    of the central molecule by accounting for the HBing observed in a
    simulation.

    :param Nc_eff: effective number of available neighbours to rotate around
    :param pbias_ave: the average biasing in hydrogen bonding donating or
        accepting over all shell neighbours
    """
    S_orient = 0
    if Nc_eff != 0:
        S_orient = np.log((Nc_eff) ** (3 / 2) * np.pi**0.5 * pbias_ave / 2)
    S_orient = max(S_orient, 0)
    return S_orient


def get_reduced_neighbours_biases(degeneracy: dict, pD_dict: dict, pA_dict: dict):
    # pylint: disable=too-many-locals
    """
    For a given labelled shell and the dictionary containing the counts for
    each neighbour type in the shell, the dictionaries for the acceptor and
    donator probabilities, use these to find the probability of accepting from
    or donating to over all other HBing to that particular neighbour type i:
    ppD_i = pD_i / (pD_i + pA_i)

    :param degeneracy: dictionary of shell constituent and count
    :param pD_dict: dictionary of neighbours donated to and how often that occurs
    :param pA_dict: dictionary of neighbours accepted from and how often that
        occurs
    """
    Nc_eff = 0
    pbiases = []
    # iterate through each neighbour and the counts for how often they occur
    # in a shell
    for i, N_i in degeneracy.items():
        # find if neighbour i is donated to or accepted from
        d = pD_dict[i] or [0, 0]
        a = pA_dict[i] or [0, 0]
        # the donor/acceptor probabilities for a given neighbour i
        pD_i = d[0]
        pA_i = a[0]
        # if the neighbour type i has both been donated to and accepted from
        # then work out probabilities of accepting from vs donating to
        if pD_i != 0 and pA_i != 0:
            # print('i', i, 'N_i', N_i, 'pA_count', round(d[1], 5),
            #     'pD_count', round(a[1], 5))
            sum_p = pD_i + pA_i
            # work out the probabilities to donate to and accept from
            # over the sum of probabilities to donate and accept
            ppD_i = pD_i / sum_p
            ppA_i = pA_i / sum_p
            # work out the effective number of available neighbour type i
            # where 0.25 is no bias in donating to/accepting from a given
            # neighbour
            N_i_eff = ppD_i * ppA_i / 0.25 * N_i
            # sum the effective neighbours for all neighbour types i
            Nc_eff += N_i_eff
            # work out the bias of donating vs accepting from neighbours of
            # type i, if both are equally likely, pbias = 0.25, if one is
            # more likely than the other the pbais < 0.25
            pbias = ppD_i * ppA_i
            # append pbiases for each occurance of neighbour type i to a
            # list to get the averages later
            # for x in range(0, N_i):
            #     pbiases.append(pbias)
            pbiases.extend([pbias] * N_i)
            # print('\t'*1, 'i', i, 'N_i', N_i, 'ppA_i', round(ppA_i, 5),
            #         'ppD_i', round(ppD_i, 5), 'N_i_eff', round(N_i_eff, 5),
            #         'pbias_i', round(pbias, 5))
        else:
            # if a neighbour type i is not both donated to AND accepted from,
            # then it is not a neighbour involved in orientational
            # motion of the central UA and instead is either an anchor
            # or not involved in hydrogen bonding.
            # for x in range(0, N_i):
            #     pbiases.append(0)
            pbiases.extend([0] * N_i)

    # find the average HB bias over all neighbours in the shell, where
    # zeros count towards the average
    pbias_ave = 0
    if len(pbiases) != 0:
        pbias_ave = sum(pbiases) / len(pbiases)

    return Nc_eff, pbias_ave


def get_donor_acceptor_probabilities(HB_vals_dict: dict, hb_selection: str):
    """
    For a given labelled shell type, find the probability of accepting from
    or donating to a given neighbour type over all other donors or acceptors

    :param vals_dict: dictionary containing the HB donors or acceptors and
        how often they occur for a given neighbour in a shell e.g.
        {"donates_to": {"labelled_donators": 0,},
        "accepts_from": {"labelled_acceptors": 0,}}
    :param degeneracy: dictionary containing the counts for each neighbour
        type in a shell
    :param hb_selection: string for determining which HB type to analyse,
        options are either "donates_to" or "accepts_from"
    """
    pHB_dict = nested_dict()
    if hb_selection in HB_vals_dict:
        total_hb_count = 0
        for c in HB_vals_dict[hb_selection].values():
            total_hb_count += c
        for hb_neighbour, count in sorted(list(HB_vals_dict[hb_selection].items())):
            pHB = get_hb_probability(count, total_hb_count)
            pHB_dict[hb_neighbour] = [pHB, count]
            # print('\t'*1, hb_selection, hb_neighbour, count,
            #         'pi_HB:', round(pHB, 4))
    return pHB_dict


def get_shell_degeneracy(shell_label):
    """
    For a given labelled shell, find the degeneracy, i.e. count of each
    unique label in the shell

    :param shell_labels: the list or tuple of the shell labels of a
        coordination shell
    """
    return Counter(shell_label)


def get_hb_probability(count: int, total_count: int):
    """
    For a given HB donor/acceptor, find the probablity of that HB occurring
    in a given unique labelled shell over all the other acceptors or donors,
    so pA_i = N_i_A / sum_i(N_i_A)
    or pD_i = N_i_D / sum_i(N_i_D)

    :param count: number of HB donors/acceptors for a given neighbour type
    :param total_count: total number of either acceptor or donor counts over
        all neighbour types
    """
    p_i = count / total_count
    return p_i
