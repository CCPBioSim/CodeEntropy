import numpy as np

# from MDAnalysis.analysis.dihedrals import Dihedral


def assign_conformation(
    data_container, dihedral, number_frames, bin_width, start, end, step
):
    """
    Create a state vector, showing the state in which the input dihedral is
    as a function of time. The function creates a histogram from the timeseries of the
    dihedral angle values and identifies points of dominant occupancy
    (called CONVEX TURNING POINTS).
    Based on the identified TPs, states are assigned to each configuration of the
    dihedral.

    Input
    -----
    dihedral_atom_group : the group of 4 atoms defining the dihedral
    number_frames : number of frames in the trajectory
    bin_width : the width of the histogram bit, default 30 degrees
    start : int, starting frame, will default to 0
    end : int, ending frame, will default to -1 (last frame in trajectory)
    step : int, spacing between frames, will default to 1

    Return
    ------
    A timeseries with integer labels describing the state at each point in time.

    """
    conformations = np.zeros(number_frames)
    phi = np.zeros(number_frames)

    # get the values of the angle for the dihedral
    # dihedral angle values have a range from -180 to 180
    for timestep in data_container.trajectory[start:end:step]:
        value = dihedral.value()
        # we want postive values in range 0 to 360 to make the peak assignment work
        # using the fact that dihedrals have circular symetry
        # (i.e. -15 degrees = +345 degrees)
        if value < 0:
            value += 360
        phi[timestep.frame] = value

    # create a histogram using numpy
    number_bins = int(360 / bin_width)
    popul, bin_edges = np.histogram(a=phi, bins=number_bins, range=(0, 360))
    bin_value = [0.5 * (bin_edges[i] + bin_edges[i + 1]) for i in range(0, len(popul))]

    # identify "convex turning-points" and populate a list of peaks
    # peak : a bin whose neighboring bins have smaller population
    # NOTE might have problems if the peak is wide with a flat or sawtooth top
    peak_values = []

    for bin_index in range(number_bins):
        # if there is no dihedrals in a bin then it cannot be a peak
        if popul[bin_index] == 0:
            pass
        # being careful of the last bin
        # (dihedrals have circular symmetry, the histogram does not)
        elif (
            bin_index == number_bins - 1
        ):  # the -1 is because the index starts with 0 not 1
            if (
                popul[bin_index] >= popul[bin_index - 1]
                and popul[bin_index] >= popul[0]
            ):
                peak_values.append(bin_value[bin_index])
        else:
            if (
                popul[bin_index] >= popul[bin_index - 1]
                and popul[bin_index] >= popul[bin_index + 1]
            ):
                peak_values.append(bin_value[bin_index])

    # go through each frame again and assign conformation state
    for frame in range(number_frames):
        # find the TP that the snapshot is least distant from
        distances = [abs(phi[frame] - peak) for peak in peak_values]
        conformations[frame] = np.argmin(distances)

    return conformations
