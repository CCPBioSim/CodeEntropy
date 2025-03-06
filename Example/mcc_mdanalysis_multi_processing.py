# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:36:57 2022

@author: bmm66251
"""

import multiprocessing as mp
import os
import sys
from datetime import datetime
from functools import partial

import MDAnalysis as mda
import numpy as nmp

from CodeEntropy.ClassCollection import BeadClasses as BC
from CodeEntropy.ClassCollection import DataContainer as DC
from CodeEntropy.ClassCollection import ModeClasses as MC
from CodeEntropy.FunctionCollection import CustomFunctions as CF
from CodeEntropy.FunctionCollection import EntropyFunctions as EF
from CodeEntropy.FunctionCollection import GeometricFunctions as GF
from CodeEntropy.FunctionCollection import UnitsAndConversions as UAC
from CodeEntropy.FunctionCollection import Utils
from CodeEntropy.IO import Writer


def rotational_axis(iBead, arg_dataContainer, arg_numFrames):
    for iFrame in range(arg_numFrames):
        # define local basis as the rotationalAxes of the first atom in the atomList of iBead
        # doesnt matter because they all have the same R and T axes
        iLocalBasis = arg_dataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

        # get the moment of inertia tensor for ibead in thid local basis
        beadMOITensor = iBead.get_moment_of_inertia_tensor_local(
            arg_localBasis=iLocalBasis, arg_frame=iFrame
        )

        # get total weighted force and torque and weigh them
        for iAtom in iBead.atomList:
            iBead.totalWeightedForces[iFrame, :] += arg_dataContainer.localForces[
                iFrame, iAtom
            ]
            iBead.totalWeightedTorques[iFrame, :] += arg_dataContainer.localTorques[
                iFrame, iAtom
            ]

        iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

        for j in range(3):
            try:
                if nmp.isclose(iBead.totalWeightedTorques[iFrame, j], 0.0):
                    # then the beadMOITensor[j,j] must be 0 as well
                    # ensure that
                    assert nmp.isclose(beadMOITensor[j, j], 0.0)
                else:
                    # inertia weight the total torque component
                    iBead.totalWeightedTorques[iFrame, j] /= nmp.sqrt(
                        beadMOITensor[j, j]
                    )
            except:
                raise AssertionError(
                    f"Moment of Intertia is non-zero for a bead lying on axis {j}"
                )
    return


def residue_UA(
    allSel,
    dataContainer,
    numFrames,
    heavyAtomArray,
    fScale,
    tScale,
    temper,
    outfile,
    resid,
):
    baseResi = resid
    baseResn = allSel.residues.resnames[resid - 1]  # again resid is 1 indexed
    baseLabel = f"{baseResn}{baseResi}"

    # create a bead collection
    nidBeadCollection = BC.BeadCollection(f"{baseLabel}_UA", dataContainer)
    nidBeadCollection.listOfBeads = []

    # add UA beads to it (a heavy atom and its bonded hydrogens make a bead)
    baseSel = allSel.select_atoms(f"resid {baseResi}")
    baseC5Idx = baseSel.select_atoms(f"name C5'").indices[0]
    baseC4Idx = baseSel.select_atoms(f"name C4'").indices[0]
    baseC3Idx = baseSel.select_atoms(f"name C3'").indices[0]

    # get all the heavy atoms and make seaparate beads from them
    heavySel = baseSel.select_atoms(f"not name H*")

    for iheavy in heavySel.indices:
        # select the atom itself and bonded hydrogen
        iua = allSel.select_atoms(
            f"index {iheavy} or (name H* and bonded index {iheavy})"
        )

        # heavy atom name
        iName = allSel.atoms.names[iheavy]

        # create a bead
        newBead = BC.Bead(
            arg_atomList=iua.indices,
            arg_hostDataContainer=dataContainer,
            arg_numFrames=numFrames,
            arg_beadName=iName,
            arg_beadResi=baseResi,
            arg_beadResn=baseResn,
            arg_beadChid="X",
        )

        newBead.position = dataContainer._labCoords[0, iheavy]
        nidBeadCollection.listOfBeads.append(newBead)
        # Utils.printflush(f"Created a bead for {iName} in {baseLabel}. Contains {newBead.get_num_atoms()} atom(s).")

    # reset weighted vectors for each bead and
    for iBead in nidBeadCollection.listOfBeads:
        iBead.reset_totalWeightedVectors((numFrames, 3))

    # reseting all matrices to zero
    nidBeadCollection.reinitialize_matrices()

    # setup translational and rotational axes
    # Translation axes : each atom is in the 5C'-4C'-3C' axes of its host base
    Utils.printflush("Assigning Translation Axes at the UA level->", end=" ")
    for iFrame in range(numFrames):
        c5coor = dataContainer._labCoords[iFrame, baseC5Idx]
        c4coor = dataContainer._labCoords[iFrame, baseC4Idx]
        c3coor = dataContainer._labCoords[iFrame, baseC3Idx]

        tAxes, tOrigin = GF.generate_orthonormal_axes_system(c5coor, c4coor, c3coor)
        dataContainer.update_translationAxesArray_at(
            arg_frame=iFrame,
            arg_atomList=baseSel.indices,
            arg_pAxes=tAxes,
            arg_orig=tOrigin,
        )

    Utils.printflush("Done")
    Utils.printflush("Assigning Rotational Axes at the UA level->", end=" ")
    for iBead in nidBeadCollection.listOfBeads:
        # get the heavy atom
        # (should only contain one)
        # can save the list in attribute to save querry time
        iheavy = list(filter(lambda idx: idx in heavyAtomArray, iBead.atomList))
        try:
            assert len(iheavy) == 1
            iheavy = iheavy[0]
        except:
            raise ValueError(
                "More than one heavy atom in an united atom bead. That is incorrect."
            )

        for iFrame in range(numFrames):
            # Rotation axes :
            # the axes will have the geometry of a
            # local spherical-polar coordinate system.
            # See our Chakravorty et. al. 2020 in JCIM

            # from each of the hydrogen atoms bonded to the heavy atom
            # get the average position lab coordinate !!! rewrite
            avgHydrogenPosition = EF.get_avg_hpos(
                arg_atom=iheavy,
                arg_frame=iFrame,
                arg_selector="all",
                arg_hostDataContainer=dataContainer,
            )

            # use the resultant vector to generate an
            # orthogonal local coordinate axes system
            # with origin at the heavy atom position
            heavyOrigin = dataContainer._labCoords[iFrame, iheavy]
            iBasis = GF.get_sphCoord_axes(arg_r=avgHydrogenPosition)

            # update rotation axes and local coords all atoms in the bead
            dataContainer.update_rotationAxesArray_at(
                arg_atomList=iBead.atomList,
                arg_frame=iFrame,
                arg_orig=heavyOrigin,
                arg_pAxes=iBasis,
            )

            dataContainer.update_localCoords(arg_atomList=iBead.atomList, arg_type="R")

    Utils.printflush("Done")

    # update local forces
    Utils.printflush("Updating Local forces->", end=" ")
    dataContainer.update_localForces(arg_type="T", arg_atomList=baseSel.indices)
    Utils.printflush("Done")

    # update torques in the arg_hostDataContainer
    Utils.printflush("Updating Local torques->", end=" ")
    for iFrame in range(numFrames):
        for iAtom in baseSel.indices:
            coords_i = dataContainer.localCoords[iFrame, iAtom]
            forces_i = dataContainer.localForces[iFrame, iAtom]
            dataContainer.localTorques[iFrame, iAtom] = CF.cross_product(
                coords_i, forces_i
            )
    Utils.printflush("Done")

    ################## COMMON OPERATIONS ######################
    # mass weighting the forces and torques
    Utils.printflush("Weighting forces and torques->", end=" ")
    for iBead in nidBeadCollection.listOfBeads:

        # mass weighting the forces for each bead (iBead) in each direction (j)
        # inertia weighting the torques for each bead (iBead) in each direction (j)

        for iFrame in range(numFrames):
            # define local basis as the rotationalAxes of the first atom in the atomList of iBead
            # doesnt matter because they all have the same R and T axes
            iLocalBasis = dataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

            # get the moment of inertia tensor for ibead in thid local basis
            beadMOITensor = iBead.get_moment_of_inertia_tensor_local(
                arg_localBasis=iLocalBasis, arg_frame=iFrame
            )

            # get total weighted force and torque and weigh them
            for iAtom in iBead.atomList:
                iBead.totalWeightedForces[iFrame, :] += dataContainer.localForces[
                    iFrame, iAtom
                ]
                iBead.totalWeightedTorques[iFrame, :] += dataContainer.localTorques[
                    iFrame, iAtom
                ]

            iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

            for j in range(3):
                try:
                    if nmp.isclose(iBead.totalWeightedTorques[iFrame, j], 0.0):
                        # then the beadMOITensor[j,j] must be 0 as well
                        # ensure that
                        assert nmp.isclose(beadMOITensor[j, j], 0.0)
                    else:
                        # inertia weight the total torque component
                        iBead.totalWeightedTorques[iFrame, j] /= nmp.sqrt(
                            beadMOITensor[j, j]
                        )
                except:
                    raise AssertionError(
                        f"Moment of Intertia is non-zero for a bead lying on axis {j}"
                    )
    Utils.printflush("Done")

    # now fill in the matrices
    Utils.printflush("Updating the submatrices ... ")
    nidBeadCollection.update_subMatrix(arg_pairString="FF", arg_verbose=3)
    nidBeadCollection.update_subMatrix(arg_pairString="TT", arg_verbose=3)
    Utils.printflush("Done")

    # make quadrant from subMatrices
    Utils.printflush("Generating Quadrants->", end=" ")
    ffQuadrant = nidBeadCollection.generate_quadrant(
        arg_pairString="FF", arg_filterZeros=0
    )
    ttQuadrant = nidBeadCollection.generate_quadrant(
        arg_pairString="TT", arg_filterZeros=0
    )
    Utils.printflush("Done")

    # scale forces/torques of these quadrants
    ffQuadrant = nmp.multiply(fScale**2, ffQuadrant)
    ttQuadrant = nmp.multiply(tScale**2, ttQuadrant)

    # remove any row or column with zero axis
    # this could have been done while generating quadrants. Can be merged if wished for
    ffQuadrant = nidBeadCollection.filter_zero_rows_columns(ffQuadrant)
    ttQuadrant = nidBeadCollection.filter_zero_rows_columns(ttQuadrant)

    # diagnolaize
    Utils.printflush("Diagonalizing->", end=" ")
    lambdasFF, eigVectorsFF = Utils.diagonalize(ffQuadrant)
    lambdasTT, eigVectorsTT = Utils.diagonalize(ttQuadrant)
    Utils.printflush("Done")

    # change to SI units
    Utils.printflush("Changing the units of eigen values to SI units->", end=" ")
    lambdasFF = UAC.change_lambda_units(lambdasFF)
    lambdasTT = UAC.change_lambda_units(lambdasTT)
    Utils.printflush("Done")

    # Create a spectrum to store these modes for
    # proper output and analyses.
    modeSpectraFF = []
    for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
        fflmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(fflmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=fflmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(fflmb, temper)
        modeSpectraFF.append(newMode)

    modeSpectraTT = []
    for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
        ttlmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(ttlmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=ttlmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(ttlmb, temper)
        modeSpectraTT.append(newMode)

    # sorting the spectrum
    Utils.printflush("Sorting spectrum in ascending order of frequencies->", end=" ")
    modeSpectraFF = MC.sort_modes(modeSpectraFF)
    modeSpectraTT = MC.sort_modes(modeSpectraTT)
    Utils.printflush("Done")

    # compute entropy
    # 1. remove the smallest 6 freqs from FF sprectrum
    #     because they may be overlapping with residue level motions
    # 2. DO NOT remove any freq from TT spectrum because
    #    they are uncoupled to any TT freq in any other hierarchy
    entropyFF = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in modeSpectraFF[6:]
    ]
    entropyTT = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in modeSpectraTT[0:]
    ]

    nidTotalEntropyFF = nmp.sum(entropyFF)
    nidTotalEntropyTT = nmp.sum(entropyTT)

    # print final outputspool.close()
    Utils.printflush("Entropy values:")

    Utils.printflush(
        "{:<40s} : {:.4f} J/mol/K".format(
            "FF Entropy (UA for {})".format(baseLabel), nidTotalEntropyFF
        )
    )
    Utils.printflush(
        "{:<40s} : {:.4f} J/mol/K".format(
            "TT Entropy (UA for {})".format(baseLabel), nidTotalEntropyTT
        )
    )
    Utils.printOut(
        outfile,
        f"UATOM {baseResn:<10}{baseResi:>5}{nidTotalEntropyFF:>12.3f}{nidTotalEntropyTT:>12.3f}",
    )

    return (nidTotalEntropyFF, nidTotalEntropyTT)


if __name__ == "__main__":
    startTime = datetime.now()
    ############## REPLACE INPUTS ##############
    wd = os.path.dirname(os.path.abspath(__file__))
    tprfile = os.path.join(wd, "data/md_A4_dna.tpr")
    trrfile = os.path.join(wd, "data/md_A4_dna_xf.trr")
    outfile = os.path.join(wd, "dna_mcc.out")
    tScale = 1
    fScale = 1
    temper = 300  # K
    pool = mp.Pool()

    # open output channel
    Writer.write_file(arg_filename=outfile)
    u = mda.Universe(tprfile, trrfile)
    dataContainer = DC.DataContainer(u)
    # number of frames
    numFrames = len(u.trajectory)
    Utils.printflush(f"Total number of frame = {numFrames}")
    mlevel = BC.BeadCollection("mlevel", dataContainer)
    allSel = dataContainer.universe.select_atoms("all")

    wholeDNABead = BC.Bead(
        arg_atomList=allSel.indices,
        arg_numFrames=numFrames,
        arg_hostDataContainer=dataContainer,
        arg_beadName="WMOL",
        arg_beadResi=0,
        arg_beadResn="WMOL",
        arg_beadChid="X",
    )
    # add the bead to the bead colleciton
    mlevel.listOfBeads = [wholeDNABead]
    Utils.printflush(
        f"Total number of beads at the whole molecule level = {len(mlevel.listOfBeads)}"
    )
    Utils.printOut(
        outfile,
        f"Total number of beads at the whole molecule level = {len(mlevel.listOfBeads)}",
    )

    # reset weighted vectors for each bead and
    # and assign a representative position
    for iBead in mlevel.listOfBeads:
        iBead.reset_totalWeightedVectors((numFrames, 3))
        iBead.position = iBead.get_center_of_mass_lab(arg_frame=0)

    # reseting all matrices to zero
    mlevel.reinitialize_matrices()

    # setup translational and rotational axes
    Utils.printflush(
        "Assigning Translation and Rotation Axes @ whole molecule level->", end=" "
    )
    # USE whole molecule's principal axes system (which changes every frame) for each atom
    allAtoms = allSel.indices
    for iFrame in range(numFrames):
        selMOI, selAxes = dataContainer.get_principal_axes(
            arg_atomList=allAtoms, arg_frame=iFrame, arg_sorted=False
        )
        selCOM = dataContainer.get_center_of_mass(
            arg_atomList=allAtoms, arg_frame=iFrame
        )

        dataContainer.update_translationAxesArray_at(
            arg_frame=iFrame, arg_atomList=allAtoms, arg_pAxes=selAxes, arg_orig=selCOM
        )
        dataContainer.update_rotationAxesArray_at(
            arg_frame=iFrame, arg_atomList=allAtoms, arg_pAxes=selAxes, arg_orig=selCOM
        )
    Utils.printflush("Done")

    # update local coordinates
    Utils.printflush("Updating Local coordinates->", end=" ")
    dataContainer.update_localCoords_of_all_atoms(arg_type="R")
    Utils.printflush("Done")

    # update local forces
    Utils.printflush("Updating Local forces->", end=" ")
    dataContainer.update_localForces_of_all_atoms(arg_type="T")
    Utils.printflush("Done")

    # update torques in the arg_hostDataContainer
    Utils.printflush("Updating Local torques->", end=" ")
    for iFrame in range(numFrames):
        for iAtom in allSel.indices:
            coords_i = dataContainer.localCoords[iFrame, iAtom]
            forces_i = dataContainer.localForces[iFrame, iAtom]
            dataContainer.localTorques[iFrame, iAtom] = CF.cross_product(
                coords_i, forces_i
            )
    Utils.printflush("Done")

    # mass weighting the forces and torques
    Utils.printflush("Weighting forces and torques->", end=" ")
    for iBead in mlevel.listOfBeads:

        # mass weighting the forces for each bead (iBead) in each direction (j)
        # inertia weighting the torques for each bead (iBead) in each direction (j)

        for iFrame in range(numFrames):
            # define local basis as the rotationalAxes of the first atom in the atomList of iBead
            # doesnt matter because they all have the same R and T axes
            iLocalBasis = dataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

            # get the moment of inertia tensor for ibead in thid local basis
            beadMOITensor = iBead.get_moment_of_inertia_tensor_local(
                arg_localBasis=iLocalBasis, arg_frame=iFrame
            )

            # get total force and torque in each direction and weight them
            for iAtom in iBead.atomList:
                iBead.totalWeightedForces[iFrame] += dataContainer.localForces[
                    iFrame, iAtom
                ]
                iBead.totalWeightedTorques[iFrame] += dataContainer.localTorques[
                    iFrame, iAtom
                ]

            iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

            for j in range(3):
                iBead.totalWeightedTorques[iFrame, j] /= nmp.sqrt(beadMOITensor[j, j])

    Utils.printflush("Done")

    # now fill in the matrices
    Utils.printflush("Updating the submatrices ... ")
    mlevel.update_subMatrix(arg_pairString="FF", arg_verbose=3)
    mlevel.update_subMatrix(arg_pairString="TT", arg_verbose=3)
    Utils.printflush("Done")

    # make quadrant from subMatrices
    # FF and TT quadrants must be symmetric
    Utils.printflush("Generating Quadrants->", end=" ")
    ffQuadrant = mlevel.generate_quadrant(arg_pairString="FF", arg_filterZeros=1)
    ttQuadrant = mlevel.generate_quadrant(arg_pairString="TT", arg_filterZeros=1)

    # scale forces/torques of these quadrants
    ffQuadrant = nmp.multiply(fScale**2, ffQuadrant)
    ttQuadrant = nmp.multiply(tScale**2, ttQuadrant)
    Utils.printflush("Done")

    # diagnolaize
    Utils.printflush("Diagonalizing->", end=" ")
    lambdasFF, eigVectorsFF = Utils.diagonalize(ffQuadrant)
    lambdasTT, eigVectorsTT = Utils.diagonalize(ttQuadrant)
    Utils.printflush("Done")

    # change to SI units
    Utils.printflush("Changing the units of eigen values to SI units->", end=" ")
    lambdasFF = UAC.change_lambda_units(lambdasFF)
    lambdasTT = UAC.change_lambda_units(lambdasTT)
    Utils.printflush("Done")

    # Create a spectrum to store these modes for
    # proper output and analyses.
    # Translational
    modeSpectraFF = []
    for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
        fflmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(fflmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=fflmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(fflmb, temper)
        modeSpectraFF.append(newMode)

    # Rotational
    modeSpectraTT = []
    for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
        ttlmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(ttlmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=ttlmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(ttlmb, temper)
        modeSpectraTT.append(newMode)

    # assign spectra to the bead collection
    mlevel.assign_attribute("modeSpectraFF", modeSpectraFF)
    mlevel.assign_attribute("modeSpectraTT", modeSpectraTT)

    # sorting the spectrum
    Utils.printflush("Sorting spectrum in ascending order of frequencies->", end=" ")
    mlevel.modeSpectraFF = MC.sort_modes(mlevel.modeSpectraFF)
    mlevel.modeSpectraTT = MC.sort_modes(mlevel.modeSpectraTT)
    Utils.printflush("Done")

    # compute entropy
    entropyFF = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in mlevel.modeSpectraFF
    ]
    entropyTT = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in mlevel.modeSpectraTT
    ]

    Utils.printflush("Entropy values:")
    Utils.printflush(
        f"{'FF Entropy (M level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K"
    )
    Utils.printOut(
        outfile, f"{'FF Entropy (M level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K"
    )

    Utils.printflush(
        f"{'TT Entropy (M level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K"
    )
    Utils.printOut(
        outfile, f"{'TT Entropy (M level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K"
    )

    ############### MOLECULE LEVEL ################

    ############## NUCLEOTIDE LEVEL ###############
    Utils.hbar(60)
    Utils.printflush(f'{"Hierarchy level. --> Nucleotide molecule <--":^60}')
    Utils.hbar(60)

    Utils.printOut(outfile, "-" * 60)
    Utils.printOut(outfile, f'{"Hierarchy level. --> Nucleotide molecule <--":^60}')
    Utils.printOut(outfile, "-" * 60)

    # number of frames
    numFrames = len(dataContainer.trajSnapshots)
    Utils.printflush(f"Total number of frame = {numFrames}")

    # Define a bead collection at the N-level
    nlevel = BC.BeadCollection("nlevel", dataContainer)
    allSel = dataContainer.universe.select_atoms("all")
    nlevel.listOfBeads = []

    # Add a bead corresponding to each residue/nucleotide to this collection
    for resid in allSel.residues.resids:
        baseResi = resid
        baseResn = allSel.residues.resnames[
            resid - 1
        ]  # -1 because residue is 1 indiced
        baseLabel = f"{baseResn}{baseResi}"
        Utils.printflush(baseLabel)

        baseSel = allSel.select_atoms(f"resid {baseResi}")

        baseBead = BC.Bead(
            arg_atomList=baseSel.indices,
            arg_numFrames=numFrames,
            arg_hostDataContainer=dataContainer,
            arg_beadName=baseLabel,
            arg_beadResi=baseResi,
            arg_beadResn=baseResn,
            arg_beadChid="X",
        )
        nlevel.listOfBeads.append(baseBead)

    Utils.printflush(
        f"Total number of beads at the Nucleotide level = {len(nlevel.listOfBeads)}"
    )
    Utils.printOut(
        outfile,
        f"Total number of beads at the Nucleotide level = {len(nlevel.listOfBeads)}",
    )

    # reset weighted vectors for each bead and
    # and assign a representative position
    for iBead in nlevel.listOfBeads:
        iBead.reset_totalWeightedVectors((numFrames, 3))
        iBead.position = iBead.get_center_of_mass_lab(arg_frame=0)

    # reseting all matrices to zero
    nlevel.reinitialize_matrices()

    # setup translational and rotational axes
    Utils.printflush("Assigning Translation Axes at Nucleotide level->", end=" ")
    # USE whole molecule's principal axes system per atom per nucleotide for translational axes
    allAtoms = allSel.indices

    for iFrame in range(numFrames):
        # compute molecular principal axes per frame
        allMOI, allPAxes = dataContainer.get_principal_axes(
            arg_atomList=allAtoms, arg_frame=iFrame, arg_sorted=False
        )
        allCOM = dataContainer.get_center_of_mass(
            arg_atomList=allAtoms, arg_frame=iFrame
        )

        dataContainer.update_translationAxesArray_at(
            arg_atomList=allAtoms, arg_frame=iFrame, arg_pAxes=allPAxes, arg_orig=allCOM
        )
    Utils.printflush("Done")

    # Use an orthogonal axes system made of  C5', C4', C3'  atoms per nucleotide for rotational axes
    Utils.printflush("Assigning Rotational Axes at Nucleotide level->", end=" ")
    for resid in allSel.residues.resids:
        baseResi = resid
        baseSel = allSel.select_atoms(f"resid {baseResi}")
        baseAtoms = baseSel.indices
        # Here you are selecting one atom so if you slice an array the shape will missmatch
        baseC5Idx = baseSel.select_atoms(f"name C5'").indices[0]
        baseC4Idx = baseSel.select_atoms(f"name C4'").indices[0]
        baseC3Idx = baseSel.select_atoms(f"name C3'").indices[0]

        for iFrame in range(numFrames):
            c5coor = dataContainer._labCoords[iFrame, baseC5Idx]
            c4coor = dataContainer._labCoords[iFrame, baseC4Idx]
            c3coor = dataContainer._labCoords[iFrame, baseC3Idx]

            ridAxes, ridOrigin = GF.generate_orthonormal_axes_system(
                c5coor, c4coor, c3coor
            )
            dataContainer.update_rotationAxesArray_at(
                arg_atomList=baseAtoms,
                arg_frame=iFrame,
                arg_orig=ridOrigin,
                arg_pAxes=ridAxes,
            )

    Utils.printflush("Done")

    # update local coordinates
    Utils.printflush("Updating Local coordinates->", end=" ")
    dataContainer.update_localCoords_of_all_atoms(arg_type="R")
    Utils.printflush("Done")

    # update local forces
    Utils.printflush("Updating Local forces->", end=" ")
    dataContainer.update_localForces_of_all_atoms(arg_type="T")
    Utils.printflush("Done")

    # update torques in the arg_hostDataContainer
    Utils.printflush("Updating Local torques->", end=" ")
    for iFrame in range(numFrames):
        for iAtom in allSel.indices:
            coords_i = dataContainer.localCoords[iFrame, iAtom]
            forces_i = dataContainer.localForces[iFrame, iAtom]
            dataContainer.localTorques[iFrame, iAtom] = CF.cross_product(
                coords_i, forces_i
            )
    Utils.printflush("Done")

    ################## COMMON OPERATIONS ######################
    # mass weighting the forces and torques
    Utils.printflush("Weighting forces and torques->", end=" ")
    for iBead in nlevel.listOfBeads:

        # mass weighting the forces for each bead (iBead) in each direction (j)
        # inertia weighting the torques for each bead (iBead) in each direction (j)

        for iFrame in range(numFrames):
            # define local basis as the rotationalAxes of the first atom in the atomList of iBead
            # doesnt matter because they all have the same R and T axes
            iLocalBasis = dataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

            # get the moment of inertia tensor for ibead in thid local basis
            beadMOITensor = iBead.get_moment_of_inertia_tensor_local(
                arg_localBasis=iLocalBasis, arg_frame=iFrame
            )

            # get total force and torque in each direction and weight them
            for iAtom in iBead.atomList:
                iBead.totalWeightedForces[iFrame] += dataContainer.localForces[
                    iFrame, iAtom
                ]
                iBead.totalWeightedTorques[iFrame] += dataContainer.localTorques[
                    iFrame, iAtom
                ]

            iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

            for j in range(3):
                iBead.totalWeightedTorques[iFrame, j] /= nmp.sqrt(beadMOITensor[j, j])

    Utils.printflush("Done")

    # now fill in the matrices
    Utils.printflush("Updating the submatrices ... ")
    nlevel.update_subMatrix(arg_pairString="FF", arg_verbose=3)
    nlevel.update_subMatrix(arg_pairString="TT", arg_verbose=3)
    Utils.printflush("Done")

    # make quadrant from subMatrices
    # FF and TT quadrants must be symmetric
    Utils.printflush("Generating Quadrants->", end=" ")
    ffQuadrant = nlevel.generate_quadrant(arg_pairString="FF", arg_filterZeros=1)
    ttQuadrant = nlevel.generate_quadrant(arg_pairString="TT", arg_filterZeros=1)

    # scale forces/torques of these quadrants
    ffQuadrant = nmp.multiply(fScale**2, ffQuadrant)
    ttQuadrant = nmp.multiply(tScale**2, ttQuadrant)
    Utils.printflush("Done")

    # diagnolaize
    Utils.printflush("Diagonalizing->", end=" ")
    lambdasFF, eigVectorsFF = Utils.diagonalize(ffQuadrant)
    lambdasTT, eigVectorsTT = Utils.diagonalize(ttQuadrant)
    Utils.printflush("Done")

    # change to SI units
    Utils.printflush("Changing the units of eigen values to SI units->", end=" ")
    lambdasFF = UAC.change_lambda_units(lambdasFF)
    lambdasTT = UAC.change_lambda_units(lambdasTT)
    Utils.printflush("Done")

    # Create a spectrum to store these modes for
    # proper output and analyses.
    # Translational
    modeSpectraFF = []
    for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
        fflmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(fflmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=fflmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(fflmb, temper)
        modeSpectraFF.append(newMode)

    # Rotational
    modeSpectraTT = []
    for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
        ttlmb, evec = mcombo
        # compute mode frequencies
        # nu = sqrt(lambda/kT)*(1/2pi)
        # Units: 1/s
        mfreq = EF.compute_frequency_from_lambda(ttlmb, temper)
        newMode = MC.Mode(
            arg_modeIdx=midx + 1,
            arg_modeEval=ttlmb,
            arg_modeEvec=evec,
            arg_modeFreq=mfreq,
        )
        newMode.modeAmpl = EF.compute_ampfac_from_lambda(ttlmb, temper)
        modeSpectraTT.append(newMode)

    # assign spectra to the bead collection
    nlevel.assign_attribute("modeSpectraFF", modeSpectraFF)
    nlevel.assign_attribute("modeSpectraTT", modeSpectraTT)

    # sorting the spectrum
    Utils.printflush("Sorting spectrum in ascending order of frequencies->", end=" ")
    nlevel.modeSpectraFF = MC.sort_modes(nlevel.modeSpectraFF)
    nlevel.modeSpectraTT = MC.sort_modes(nlevel.modeSpectraTT)
    Utils.printflush("Done")

    # compute entropy
    entropyFF = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper)
        for m in nlevel.modeSpectraFF[6:]
    ]
    entropyTT = [
        EF.calculate_entropy_per_dof(m.modeFreq, temper)
        for m in nlevel.modeSpectraTT[0:]
    ]

    Utils.printflush("Entropy values:")
    Utils.printflush(
        f"{'FF Entropy (N level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K"
    )
    Utils.printOut(
        outfile, f"{'FF Entropy (N level)':<40s} : {nmp.sum(entropyFF):.4f} J/mol/K"
    )

    Utils.printflush(
        f"{'TT Entropy (N level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K"
    )
    Utils.printOut(
        outfile, f"{'TT Entropy (N level)':<40s} : {nmp.sum(entropyTT):.4f} J/mol/K"
    )

    ############### NUCLEOTIDE LEVEL ##################

    ############### UNITED ATOM LEVEL ##################
    Utils.hbar(60)
    Utils.printflush(f'{"Hierarchy level. --> United Atom <--":^60}')
    Utils.hbar(60)

    Utils.printOut(outfile, "-" * 60)
    Utils.printOut(outfile, f'{"Hierarchy level. --> United Atom <--":^60}')
    Utils.printOut(outfile, "-" * 60)

    # number of frames
    numFrames = len(dataContainer.trajSnapshots)
    Utils.printflush(f"Total number of frame = {numFrames}")
    allSel = dataContainer.universe.select_atoms("all")

    # Heavy atom array
    heavyAtomArray = allSel.select_atoms("not name H*").indices

    f = partial(
        residue_UA, allSel, dataContainer, numFrames, heavyAtomArray, 1, 1, 300, outfile
    )
    items = allSel.residues.resids
    result = pool.map(f, items)
    pool.close()
    pool.join()
    result = nmp.sum(nmp.array(result), axis=0)
    totalUAEntropyFF = result[0]
    totalUAEntropyTT = result[1]

    # initialize total entropy values specific to this level
    # totalUAEntropyFF = 0
    # totalUAEntropyTT = 0

    # for each nucleotide, entropy at the UA level will be computed.
    # That is why, a loop is set to go through each nucleotide, get its UA beads,
    # and entropy is computed from that.
    # for resid in allSel.residues.resids:
    #     baseResi = resid
    #     baseResn = allSel.residues.resnames[resid - 1] #again resid is 1 indexed
    #     baseLabel = f"{baseResn}{baseResi}"

    #     # create a bead collection
    #     nidBeadCollection = BC.BeadCollection(f"{baseLabel}_UA", dataContainer)
    #     nidBeadCollection.listOfBeads = []

    #     # add UA beads to it (a heavy atom and its bonded hydrogens make a bead)
    #     baseSel = allSel.select_atoms(f"resid {baseResi}")
    #     baseC5Idx = baseSel.select_atoms(f"name C5'").indices[0]
    #     baseC4Idx = baseSel.select_atoms(f"name C4'").indices[0]
    #     baseC3Idx = baseSel.select_atoms(f"name C3'").indices[0]

    #     # get all the heavy atoms and make seaparate beads from them
    #     heavySel = baseSel.select_atoms(f"not name H*")

    #     for iheavy in heavySel.indices:
    #         # select the atom itself and bonded hydrogen
    #         iua = allSel.select_atoms(f"index {iheavy} or (name H* and bonded index {iheavy})")

    #         # heavy atom name
    #         iName = allSel.atoms.names[iheavy]

    #         # create a bead
    #         newBead = BC.Bead(arg_atomList=iua.indices,\
    #             arg_hostDataContainer=dataContainer,\
    #             arg_numFrames=numFrames,\
    #             arg_beadName = iName,\
    #             arg_beadResi = baseResi,\
    #             arg_beadResn = baseResn,\
    #             arg_beadChid = "X")

    #         newBead.position = dataContainer._labCoords[0, iheavy]
    #         nidBeadCollection.listOfBeads.append(newBead)
    #         # Utils.printflush(f"Created a bead for {iName} in {baseLabel}. Contains {newBead.get_num_atoms()} atom(s).")

    #     # reset weighted vectors for each bead and
    #     for iBead in nidBeadCollection.listOfBeads:
    #         iBead.reset_totalWeightedVectors( (numFrames,3) )

    #     # reseting all matrices to zero
    #     nidBeadCollection.reinitialize_matrices()

    #     # setup translational and rotational axes
    #     # Translation axes : each atom is in the 5C'-4C'-3C' axes of its host base
    #     Utils.printflush("Assigning Translation Axes at the UA level->", end = ' ')
    #     for iFrame in range(numFrames):
    #         c5coor = dataContainer._labCoords[iFrame, baseC5Idx]
    #         c4coor = dataContainer._labCoords[iFrame, baseC4Idx]
    #         c3coor = dataContainer._labCoords[iFrame, baseC3Idx]

    #         tAxes, tOrigin = GF.generate_orthonormal_axes_system(c5coor, c4coor, c3coor)
    #         dataContainer.update_translationAxesArray_at(arg_frame= iFrame, \
    #                                                      arg_atomList= baseSel.indices, \
    #                                                      arg_pAxes=tAxes, \
    #                                                      arg_orig=tOrigin)

    #     Utils.printflush('Done')
    #     Utils.printflush("Assigning Rotational Axes at the UA level->", end = ' ')
    #     for iBead in nidBeadCollection.listOfBeads:
    #         # get the heavy atom
    #         # (should only contain one)
    #         # can save the list in attribute to save querry time
    #         iheavy = list(filter(lambda idx: idx in heavyAtomArray, iBead.atomList))
    #         try:
    #             assert(len(iheavy) == 1)
    #             iheavy = iheavy[0]
    #         except:
    #             raise ValueError("More than one heavy atom in an united atom bead. That is incorrect.")

    #         for iFrame in range(numFrames):
    #             # Rotation axes :
    #             # the axes will have the geometry of a
    #             # local spherical-polar coordinate system.
    #             # See our Chakravorty et. al. 2020 in JCIM

    #             # from each of the hydrogen atoms bonded to the heavy atom
    #             # get the average position lab coordinate !!! rewrite
    #             avgHydrogenPosition = EF.get_avg_hpos(arg_atom= iheavy, \
    #                 arg_frame = iFrame, \
    #                 arg_selector = "all", \
    #                 arg_hostDataContainer = dataContainer)

    #             # use the resultant vector to generate an
    #             # orthogonal local coordinate axes system
    #             # with origin at the heavy atom position
    #             heavyOrigin = dataContainer._labCoords[iFrame, iheavy]
    #             iBasis = GF.get_sphCoord_axes(arg_r=avgHydrogenPosition)

    #             # update rotation axes and local coords all atoms in the bead
    #             dataContainer.update_rotationAxesArray_at(arg_atomList= iBead.atomList, \
    #                                                       arg_frame= iFrame, \
    #                                                       arg_orig = heavyOrigin, \
    #                                                       arg_pAxes= iBasis)

    #             dataContainer.update_localCoords(arg_atomList=iBead.atomList, arg_type="R")

    #     Utils.printflush('Done')

    #     # update local forces
    #     Utils.printflush("Updating Local forces->", end = ' ' )
    #     dataContainer.update_localForces(arg_type="T", arg_atomList=baseSel.indices)
    #     Utils.printflush('Done')

    #     #update torques in the arg_hostDataContainer
    #     Utils.printflush("Updating Local torques->", end = ' ')
    #     for iFrame in range(numFrames):
    #         for iAtom in baseSel.indices:
    #             coords_i = dataContainer.localCoords[iFrame, iAtom]
    #             forces_i = dataContainer.localForces[iFrame, iAtom]
    #             dataContainer.localTorques[iFrame,iAtom] = CF.cross_product(coords_i,forces_i)
    #     Utils.printflush('Done')

    #     ################## COMMON OPERATIONS ######################
    #     # mass weighting the forces and torques
    #     Utils.printflush("Weighting forces and torques->", end=' ')
    #     for iBead in nidBeadCollection.listOfBeads:

    #         # mass weighting the forces for each bead (iBead) in each direction (j)
    #         # inertia weighting the torques for each bead (iBead) in each direction (j)

    #         for iFrame in range(numFrames):
    #             # define local basis as the rotationalAxes of the first atom in the atomList of iBead
    #             # doesnt matter because they all have the same R and T axes
    #             iLocalBasis = dataContainer.rotationAxesArray[iFrame][iBead.atomList[0]]

    #             #get the moment of inertia tensor for ibead in thid local basis
    #             beadMOITensor = iBead.get_moment_of_inertia_tensor_local(arg_localBasis = iLocalBasis, \
    #                                                                       arg_frame = iFrame)

    #             # get total weighted force and torque and weigh them
    #             for iAtom in iBead.atomList:
    #                 iBead.totalWeightedForces[iFrame,:] += dataContainer.localForces[iFrame, iAtom]
    #                 iBead.totalWeightedTorques[iFrame,:] += dataContainer.localTorques[iFrame, iAtom]

    #             iBead.totalWeightedForces[iFrame] /= nmp.sqrt(iBead.get_total_mass())

    #             for j in range(3):
    #                 try:
    #                     if nmp.isclose(iBead.totalWeightedTorques[iFrame,j] , 0.0):
    #                         # then the beadMOITensor[j,j] must be 0 as well
    #                         # ensure that
    #                         assert(nmp.isclose(beadMOITensor[j,j] , 0.0))
    #                     else:
    #                         # inertia weight the total torque component
    #                         iBead.totalWeightedTorques[iFrame,j] /= nmp.sqrt(beadMOITensor[j,j])
    #                 except:
    #                     raise AssertionError(f"Moment of Intertia is non-zero for a bead lying on axis {j}")
    #     Utils.printflush('Done')

    #     # now fill in the matrices
    #     Utils.printflush("Updating the submatrices ... ")
    #     nidBeadCollection.update_subMatrix(arg_pairString="FF",arg_verbose=3)
    #     nidBeadCollection.update_subMatrix(arg_pairString="TT",arg_verbose=3)
    #     Utils.printflush('Done')

    #     #make quadrant from subMatrices
    #     Utils.printflush("Generating Quadrants->",end = ' ')
    #     ffQuadrant = nidBeadCollection.generate_quadrant(arg_pairString="FF",arg_filterZeros=0)
    #     ttQuadrant = nidBeadCollection.generate_quadrant(arg_pairString="TT",arg_filterZeros=0)
    #     Utils.printflush("Done")

    #     # scale forces/torques of these quadrants
    #     ffQuadrant = nmp.multiply(fScale**2, ffQuadrant)
    #     ttQuadrant = nmp.multiply(tScale**2, ttQuadrant)

    #     # remove any row or column with zero axis
    #     # this could have been done while generating quadrants. Can be merged if wished for
    #     ffQuadrant = nidBeadCollection.filter_zero_rows_columns(ffQuadrant)
    #     ttQuadrant = nidBeadCollection.filter_zero_rows_columns(ttQuadrant)

    #     #diagnolaize
    #     Utils.printflush("Diagonalizing->", end = ' ')
    #     lambdasFF, eigVectorsFF  = Utils.diagonalize(ffQuadrant)
    #     lambdasTT, eigVectorsTT  = Utils.diagonalize(ttQuadrant)
    #     Utils.printflush('Done')

    #     # change to SI units
    #     Utils.printflush('Changing the units of eigen values to SI units->', end = ' ')
    #     lambdasFF = UAC.change_lambda_units(lambdasFF)
    #     lambdasTT = UAC.change_lambda_units(lambdasTT)
    #     Utils.printflush('Done')

    #     # Create a spectrum to store these modes for
    #     # proper output and analyses.
    #     modeSpectraFF = []
    #     for midx, mcombo in enumerate(zip(lambdasFF, eigVectorsFF)):
    #         fflmb, evec = mcombo
    #         # compute mode frequencies
    #         # nu = sqrt(lambda/kT)*(1/2pi)
    #         # Units: 1/s
    #         mfreq = EF.compute_frequency_from_lambda(fflmb, temper)
    #         newMode = MC.Mode(arg_modeIdx = midx + 1, \
    #             arg_modeEval = fflmb, \
    #             arg_modeEvec = evec, \
    #             arg_modeFreq = mfreq)
    #         newMode.modeAmpl = EF.compute_ampfac_from_lambda(fflmb, temper)
    #         modeSpectraFF.append(newMode)

    #     modeSpectraTT = []
    #     for midx, mcombo in enumerate(zip(lambdasTT, eigVectorsTT)):
    #         ttlmb, evec = mcombo
    #         # compute mode frequencies
    #         # nu = sqrt(lambda/kT)*(1/2pi)
    #         # Units: 1/s
    #         mfreq = EF.compute_frequency_from_lambda(ttlmb, temper)
    #         newMode = MC.Mode(arg_modeIdx = midx + 1, \
    #             arg_modeEval = ttlmb, \
    #             arg_modeEvec = evec, \
    #             arg_modeFreq = mfreq)
    #         newMode.modeAmpl = EF.compute_ampfac_from_lambda(ttlmb, temper)
    #         modeSpectraTT.append(newMode)

    #     # sorting the spectrum
    #     Utils.printflush('Sorting spectrum in ascending order of frequencies->', end = ' ')
    #     modeSpectraFF = MC.sort_modes(modeSpectraFF)
    #     modeSpectraTT = MC.sort_modes(modeSpectraTT)
    #     Utils.printflush('Done')

    #     # compute entropy
    #     # 1. remove the smallest 6 freqs from FF sprectrum
    #     #     because they may be overlapping with residue level motions
    #     # 2. DO NOT remove any freq from TT spectrum because
    #     #    they are uncoupled to any TT freq in any other hierarchy
    #     entropyFF = [EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in modeSpectraFF[6:]]
    #     entropyTT = [EF.calculate_entropy_per_dof(m.modeFreq, temper) for m in modeSpectraTT[0:]]

    #     nidTotalEntropyFF = nmp.sum(entropyFF)
    #     nidTotalEntropyTT = nmp.sum(entropyTT)

    #     # print final outputs
    #     Utils.printflush("Entropy values:")

    #     Utils.printflush('{:<40s} : {:.4f} J/mol/K'.format('FF Entropy (UA for {})'.format(baseLabel), nidTotalEntropyFF))
    #     Utils.printflush('{:<40s} : {:.4f} J/mol/K'.format('TT Entropy (UA for {})'.format(baseLabel), nidTotalEntropyTT))
    #     Utils.printOut(outfile,f'UATOM {baseResn:<10}{baseResi:>5}{nidTotalEntropyFF:>12.3f}{nidTotalEntropyTT:>12.3f}')

    #     totalUAEntropyFF += nidTotalEntropyFF
    #     totalUAEntropyTT += nidTotalEntropyTT

    # Final information
    Utils.hbar(60)
    Utils.printflush(
        f"{'Total Entropy FF (UA level)':<25} : {totalUAEntropyFF:>15.3f} J/mol/K"
    )
    Utils.printflush(
        f"{'Total Entropy TT (UA level)':<25} : {totalUAEntropyTT:>15.3f} J/mol/K"
    )
    # Utils.hbar(60)

    Utils.printOut(outfile, "_" * 60)
    Utils.printOut(
        outfile,
        f"{'Total Entropy FF (UA level)':<25} : {totalUAEntropyFF:>15.3f} J/mol/K",
    )
    Utils.printOut(
        outfile,
        f"{'Total Entropy TT (UA level)':<25} : {totalUAEntropyTT:>15.3f} J/mol/K",
    )
    Utils.printOut(outfile, "-" * 60)
    print(datetime.now() - startTime)
    ############### UNITED ATOM LEVEL ##################
# END
