#!/usr/bin/env python

import sys
import logging


import math
import numpy as np
from numpy import linalg as LA

from collections import defaultdict
from collections import Counter
nested_dict = lambda: defaultdict(nested_dict) 
        #create nested dict in one go



def processEE(num_frames, totFrames, Aclass, solvent, waterTuple, 
        temperature, level, name, forceUnits, verbosePrint):
    '''
    Populate and process Smix and Sor into global dictionaries
    '''

    verbosePrint('\n\n---ANALYSIS_TYPE_%s---\n' % (name))
    verbosePrint('\n\n---ORIENTATIONAL_ENTROPY_CALCULATIONS---\n')
    if any(i in solvent for i in waterTuple) == True:
        SorCalculation(Aclass, 'Sor_test2', level, waterTuple, verbosePrint)


    verbosePrint('\n\n --UA_VIBRATIONAL_ENTROPY--\n')
    SvibCalculations(totFrames, Aclass, temperature, forceUnits, verbosePrint)
    verbosePrint('\n\n --WM_VIBRATIONAL_ENTROPY--\n')
    WMSvibCalculations(Aclass, totFrames, temperature, 
            forceUnits, verbosePrint)

    verbosePrint('\n\n --WM_CONFORMATIONAL_ENTROPY(DIH)--\n')
    process_dihedrals(Aclass, verbosePrint)


    verbosePrint('\n\n--PRINT_ALL_VARIABLES--\n')
    printAllVariables(num_frames, Aclass, level, name, solvent, 
            waterTuple, verbosePrint)


    if level in ['residLevel_resname', 'atomLevel']:
        verbosePrint('\n\n--CONTACTS--\n')
        contactCalculation(Aclass, level, totFrames, verbosePrint)



def contactCalculation(Aclass, level, totFrames, verbosePrint):
    '''
    resid matrix of RAD contacts between resids that are not water and 
    the nearest non-like to it.
    '''

    verbosePrint('\n*******_RESID_CONTACTS_**********\n')
    data = open('resid_contact_matrix_%s.csv' % (level), 'w')
    if level == 'atomLevel':
        data.write('\n'.join(['centre_resid,neighbour_resid,count,'\
                'centre_resname,neighbour_resname,centre_atom,'\
                'neighbour_atom']) 
                + '\n')   
    else:
        data.write('\n'.join(['centre_resid,neighbour_resid,count,'\
                'centre_resname,neighbour_resname']) 
                + '\n')   
    for centre_info, neighbour_resid_key in \
            sorted(list(Aclass.resid_contact_matrix_dict.items())):
        for neighbour_info, count in \
                sorted(list(neighbour_resid_key.items())):
            #verbosePrint(centre_info, neighbour_info, count)
            if count > 0:
                if level == 'atomLevel':
                    centre_atom = centre_info[0]
                    centre_resname = centre_info[1]
                    centre_resid = centre_info[2]
                    neighbour_atom = neighbour_info[0]
                    neighbour_resname = neighbour_info[1]
                    neighbour_resid = neighbour_info[2]
                    count = float(count) / float(totFrames)
                    data.write('\n'.join(['%s,%s,%s,%s,%s,%s,%s' 
                            % (centre_resid, 
                            neighbour_resid, count, centre_resname, 
                            neighbour_resname, centre_atom, 
                            neighbour_atom)]) + '\n')

                else:
                    centre_resname = centre_info[0]
                    centre_resid = centre_info[1]
                    neighbour_resname = neighbour_info[0]
                    neighbour_resid = neighbour_info[1]
                    count = float(count) / float(totFrames)
                    data.write('\n'.join(['%s,%s,%s,%s,%s' % (centre_resid, 
                            neighbour_resid, count, centre_resname, 
                            neighbour_resname)]) + '\n')
            else:
                continue


    data.close()



def SorCalculation(Aclass, methodType, level, waterTuple, verbosePrint):
    '''
    '''

    por_dict = nested_dict()
    shells_ppD_ppA_dict = nested_dict()
    verbosePrint('\n\n__ORIENTATIONAL_ENTROPY_%s' % (methodType))
    num_molecules0 = sum(count
            for atomType in Aclass.RADshell_Sor_dict.values()
            #for Nc in atomType.values() 
            for RADshell_num in atomType.values() 
            for RAD_str in RADshell_num.values()
            for ADs in RAD_str.values()
            for key, HBcount in ADs.items() 
            for HB, count in HBcount.items()
            if key == 'D')
    verbosePrint('\nall molecule count:', num_molecules0)
    S1 = 0
    for nearest, assigned_key in \
            sorted(list(Aclass.RADshell_Sor_dict.items())):
        num_molecules1 = sum(count
                #for Nc in assigned_key.values() 
                for RADshell_num in assigned_key.values() 
                for RAD_str in RADshell_num.values()
                for ADs in RAD_str.values()
                for AD, HBcount in ADs.items() 
                for HB, count in HBcount.items()
                if AD == 'D')

        nearest_resname = nearest.split('_')[0]
        verbosePrint('\n\nnearest:', nearest, 'count:', num_molecules1, 
                '/', num_molecules0)

        S2 = 0
        for assigned, shell_num_key in sorted(list(assigned_key.items())):
            num_molecules2 = sum(count
                    #for RADshell_num in Nc_key.values() 
                    for RAD_str in shell_num_key.values()
                    for ADs in RAD_str.values()
                    for AD, HBcount in ADs.items() 
                    for HB, count in HBcount.items()
                    if AD == 'D')

            assigned_resname = assigned[0].split('_')[0]
            verbosePrint('\tassigned:', assigned, 'count:', num_molecules2,
                    '/', num_molecules1)
            N_H = assigned[1]
            verbosePrint('\tN_H', N_H)
            S4 = 0
            for shell_num, RADlist_key in \
                    sorted(list(shell_num_key.items())):

                num_molecules4 = sum(count
                        for ADs in RADlist_key.values()
                        for AD, HBcount in ADs.items() 
                        for HB, count in HBcount.items()
                        if AD == 'D')

                verbosePrint('\t\t\tshell_num:', shell_num[0], 'count:', 
                        num_molecules4, '/', num_molecules2)
                S5 = 0
                S_shell = 0
                ave_pbias_ave = 0
                ave_Nc_eff = 0

                for RADlist, AD_key in \
                        sorted(list(RADlist_key.items())):
                    verbosePrint('\t'*4, '-----RADshell------')
                    num_molecules5 = sum(count
                            for AD, HBcount in AD_key.items() 
                            for HB, count in HBcount.items()
                            if AD == 'D')

                    ## dict for p(c) log p(c)
                    if methodType == 'Sor_test2' and \
                            level == 'moleculeLevel' and \
                            assigned[0] in waterTuple and \
                            int(shell_num[0]) == 1:
                        newRADlist = []
                        for n in RADlist:
                            if n == '0':
                                newRADlist.append(nearest_resname)
                            else:
                                newRADlist.append(n)
                        newRADlist = tuple(sorted(newRADlist))
                        try:
                            if newRADlist not in \
                                    Aclass.p_coord_dict[len(RADlist)]:
                                Aclass.p_coord_dict[len(RADlist)]\
                                        [newRADlist] = 0
                            Aclass.p_coord_dict[len(RADlist)][newRADlist] += \
                                    num_molecules5
                        except AttributeError:
                            pass

                    RAD_l = ('%s' % (','.join(map(str, RADlist))))
                    verbosePrint('\t\t\t\tneighbours:', RAD_l, 'Nc:', 
                            len(RADlist), 'count:', 
                            num_molecules5, '/', num_molecules4)
                    S6 = 0
                    Sor_list = []
                    pbias_ave = 0
                    Nc_eff = 0


                    if methodType == 'Sor_test2':
                        D_values = None
                        A_values = None
                        for AD, values in \
                                sorted(list(AD_key.items())):
                            if AD == 'D':
                                D_values = values
                            if AD == 'A':
                                A_values = values
                        verbosePrint('\t'*4, '* SEPARATE ADs *')
                        S7, pbias_ave, Nc_eff, ppD_ppA_dict = \
                                RAD_or_entropyCalc_test2(
                                RADlist, D_values,
                                A_values, N_H, shell_num[1], 
                                num_molecules5, nearest_resname, 
                                waterTuple, verbosePrint)

                        ### ppD_ppA_dict population
                        for c in range(0, int(num_molecules5)):
                            for i, pp in ppD_ppA_dict.items():
                                ppD = pp[0]
                                ppA = pp[1]
                                if i not in shells_ppD_ppA_dict[nearest]\
                                        [assigned]\
                                        [float(shell_num[0])]:
                                    shells_ppD_ppA_dict[nearest][assigned]\
                                            [float(shell_num[0])]\
                                            [i] = [0, 0, 0]
                                ppD_stored = shells_ppD_ppA_dict[nearest]\
                                        [assigned][float(shell_num[0])][i][0]
                                ppA_stored = shells_ppD_ppA_dict[nearest]\
                                        [assigned][float(shell_num[0])][i][1]
                                count_stored = shells_ppD_ppA_dict[nearest]\
                                        [assigned][float(shell_num[0])][i][2]


                                count_add = count_stored + 1
                                ppD_add = ((ppD + ppD_stored * 
                                        count_stored) / float(count_add))
                                ppA_add = ((ppA + ppA_stored * 
                                        count_stored) / float(count_add))

                                shells_ppD_ppA_dict[nearest][assigned]\
                                        [float(shell_num[0])]\
                                        [i][0] = ppD_add
                                shells_ppD_ppA_dict[nearest][assigned]\
                                        [float(shell_num[0])]\
                                        [i][1] = ppA_add
                                shells_ppD_ppA_dict[nearest][assigned]\
                                        [float(shell_num[0])]\
                                        [i][2] = count_add


                                            
                        Sor_list.append([S7, 'DA'])
                        verbosePrint('\t\t\t\tS7_DA = ', round(S7, 3))


                    Sor_list = sorted(Sor_list)
                    verbosePrint()
                    verbosePrint('\t'*4, '*** Sor-list:')
                    c = 0
                    for s, label in Sor_list:
                        c += 1
                        if c == 1:
                            verbosePrint('\t'*4, round(s, 3), label, 
                                    '<-- selected Sor')
                        else:
                            verbosePrint('\t'*4, round(s, 3), label)
                    #### use smallest Sor as selected value
                    S6 += (Sor_list[0][0] * float(num_molecules5) / 
                        float(num_molecules4))
                    S_shell += (Sor_list[0][0] * 
                            float(num_molecules5) / 
                            float(num_molecules4))
                    ave_pbias_ave += (pbias_ave * 
                            float(num_molecules5) / 
                            float(num_molecules4))

                    ave_Nc_eff += (Nc_eff * 
                            float(num_molecules5) / 
                            float(num_molecules4))

                    if methodType == 'Sor_test2':
                        Aclass.Sor_reference_dict[nearest]\
                                [assigned[0]]\
                                [str(shell_num[0])][RADlist] = \
                                Sor_list[0][0] * 8.314
                        #print(RADlist, Sor_list[0][0] * 8.314)

                    verbosePrint('\t\t\t\tS6 = ', round(S6, 3))
                    verbosePrint()
                    S5 += (S6 * float(num_molecules4) /
                    float(num_molecules2))




                Aclass.allVariables_dict[nearest][assigned]\
                        [float(shell_num[0])]['%s' % (methodType)] = \
                        S_shell * 8.314, num_molecules4

                '''
                if methodType == 'Sor_test2':
                    Aclass.allVariables_dict[nearest][assigned]\
                            [float(shell_num[0])]\
                            ['pbias_ave_%s' % (methodType)] = \
                            ave_pbias_ave, num_molecules4

                    Aclass.allVariables_dict[nearest][assigned]\
                            [float(shell_num[0])]\
                            ['Nc_eff_%s' % (methodType)] = \
                            ave_Nc_eff, num_molecules4
                '''

                verbosePrint('\t\t\tS5 = ', round(S5, 3))
                S4 += (S5 * float(num_molecules4) /
                float(num_molecules2))

            verbosePrint('\t\tS4 = ', round(S4, 3))
            S2 += (S4 * float(num_molecules1) /
            float(num_molecules0))

        verbosePrint('\tS2 = ', round(S2, 3))
        S1 += S2
    verbosePrint('S1 = ', round(S1, 3))


    if methodType == 'Sor_test2':
        verbosePrint('\n\n__ppX_ORIENTATIONAL_ENTROPY_%s' % (methodType))
        verbosePrint('nearest,assigned,shell_num,i,ppD,ppA,count')
        for nearest, assigned_key in sorted(list(shells_ppD_ppA_dict.items())):
            for assigned, shell_num_key in sorted(list(assigned_key.items())):
                for shell_num, i_key in sorted(list(shell_num_key.items())):
                    for i, pp in sorted(list(i_key.items())):
                        ppD, ppA, count = pp[0], pp[1], pp[2]
                        verbosePrint('%s,%s,%s,%s,%s,%s,%s' % 
                                (nearest, assigned[0], int(shell_num), 
                                i, round(ppD, 5), round(ppA, 5), count))




def RAD_or_entropyCalc_test2(shell, shellD, shellA, N_H, shell_num, 
        referenceCount, nearest, waterTuple, verbosePrint):
    '''
    ***** S_or
    '''

    ####### get info about shell
    shellCount = Counter([str(N) for N in shell]) 
        #count constituents in shell
    Nc = sum(shellCount.values())
    Nc_solvent_only = 0
    for n in shell:
        if n == 'X':
            Nc_solvent_only += 1
        else:
            try:
                n = int(n)
                if n != 0:
                    Nc_solvent_only += 1
                else:
                    continue
            except ValueError:
                pass


    soluteA_pi, soluteD_pi = None, None
    pA_degen_dict, pD_degen_dict = None, None
    for AD, shellDs in [['A', shellA], ['D', shellD]]:
        verbosePrint('\t'*5, '__%s__' % (AD))
        ####### get info about shell
        shellCount = Counter([str(N) for N in shell]) 
            #count constituents in shell
        Dcount = Counter([tuple(Ds) for Ds in shellDs])
        num_orients = len(shellDs)

        solute_donor = False
        solvent_donor = False

        for ds in shellDs:
            for d in ds:
                try:
                    d = int(d)
                    if d == 0:
                        solute_donor = True
                    else:
                        continue
                except ValueError:
                    if d in ['X', 'H']:
                        solvent_donor = True
                    else:
                        solute_donor = True


        ####### pi only, degeneracy info for each orientation type observed
        i_counts_dict = {}
        tot_i_count = 0
        for donors, count in Dcount.items():
            for i in donors:
                if i not in i_counts_dict:
                    i_counts_dict[i]  = 0
                i_counts_dict[i] += count
                tot_i_count += count

        verbosePrint('\t'*5, '--i--')
        for i, count in i_counts_dict.items():
            verbosePrint('\t'*5, i, count)
        verbosePrint('\t'*5, 'sum i counts:', tot_i_count)

        pi_degen_dict = {}
        solute_pis = 0
        largest_pi = 0
        ww_Ds = 0
        for i, count in i_counts_dict.items():
            degeneracy = shellCount[i]
            if degeneracy == 0:
                degeneracy = 1
            pi = (float(count) / float(tot_i_count)) / float(degeneracy)
            pi_degen_dict[i] = [pi, degeneracy, count]
            verbosePrint('\t'*6, 'pi', i, round(pi, 4), 
                    'degen:', degeneracy, 
                    'pi_d:', round((pi * degeneracy), 4))
            #verbosePrint(i, pi)
            if pi * degeneracy > largest_pi:
                largest_pi = pi * degeneracy

            solute_pi, solvent_pi = False, False
            try:
                i = int(i)
                if i == 0 and nearest not in waterTuple:
                    solute_pi = True
                else:
                    ww_Ds += count
                    continue
            except ValueError:
                if i in ['X', 'H']:
                    solvent_pi = True
                    ww_Ds += count
                else:
                    solute_pi = True
            if solute_pi == True:
                solute_pis += pi * degeneracy



        ####### calculate number of observed orientations
        #pi_max = max(map(lambda x: x[0], pi_degen_dict.values()))
        #verbosePrint('\t'*6, 'pi_max', round(pi_max, 5))

        #verbosePrint(ww_Ds, referenceCount)
        per_w_Ds = ww_Ds / float(referenceCount)


        if AD == 'A':
            pA_degen_dict = pi_degen_dict
        if AD == 'D':
            pD_degen_dict = pi_degen_dict

        ####### pij, degeneracy info for each orientation type observed
        verbosePrint('\t'*5, '--ij--')
        pij_degen_dict = {}
        for donors, count in Dcount.items():
            verbosePrint('\t'*5, donors, count)
            donor_constituents = Counter([str(dc) for dc in donors])
            degeneracy = 1
            sum_weight = 0
            for dc, count2 in donor_constituents.items():
                if count2 == N_H and N_H != 1:
                    degeneracy = 0
                    for num in range(shellCount[dc]-(N_H-1), 0, -1):
                        degeneracy += num
                if N_H == 1:
                    degeneracy = shellCount[dc]
                if count2 != N_H and N_H != 1:
                    degeneracy *= shellCount[dc]

            if degeneracy == 0:
                degeneracy = 1
            pij = (float(count) / float(num_orients)) / float(degeneracy)
            pij_degen_dict[donors] = [pij, degeneracy, sum_weight]

        verbosePrint('\t'*5, 'sum pair counts:', num_orients)



        solute_pijs = 0
        largest_pij = 0
        for donors, pij in pij_degen_dict.items():
            verbosePrint('\t'*6, 'pij', donors, round(pij[0], 4), 
                    'degen:', pij[1], 
                    'pij_d:', round((pij[0] * pij[1]), 4))
            solute_pij, solvent_pij = False, False
            for d in donors:
                try:
                    d = int(d)
                    if d == 0:
                        solute_pij = True
                    else:
                        continue
                except ValueError:
                    if d in ['X', 'H']:
                        solvent_pij = True
                    else:
                        solute_pij = True
            if solute_pij == True:
                solute_pijs += pij[0] * pij[1]

            if pij[0] * pij[1] > largest_pij:
                largest_pij = pij[0] * pij[1]


        ####### calculate number of observed orientations
        pij_max = max(map(lambda x: x[0], pij_degen_dict.values()))
        verbosePrint('\t'*6, 'pij_max', round(pij_max, 5))

        por_list = []
        omega_obs = 0
        for donors, pij in pij_degen_dict.items():
            om = (float(pij[0]) / float(pij_max)) * pij[1]
            por_list.append((donors, om))
            omega_obs += om


        if AD == 'D':
            soluteD_pi = solute_pis
        if AD == 'A':
            soluteA_pi = solute_pis


        verbosePrint('\t'*6, 'solute_donor', solute_donor)
        verbosePrint('\t'*6, 'Nc, Nc_solvent_only', Nc, Nc_solvent_only)
        verbosePrint('\t'*6, 'solute_pijs', round(solute_pijs, 3))
        verbosePrint('\t'*6, 'solute_pis', round(solute_pis, 3))
        verbosePrint('\t'*6, 'largest_pij', round(largest_pij, 3))
        verbosePrint('\t'*6, 'largest_pi', round(largest_pi, 3))




    S = 0
    ppD_ppA_dict = {}
    pbias_ave = 0
    Nc_eff = 0
    if Nc != 0:
        pbias_list = []
        for i, N_i in shellCount.items():
            pD_i, pA_i = 0, 0
            pD_count, pA_count, pbias = 0, 0, 0
            if i in pA_degen_dict:
                pA_i = pA_degen_dict[i][0] * pA_degen_dict[i][1]
                pA_count = pA_degen_dict[i][2]

            if i in pD_degen_dict:
                pD_i = pD_degen_dict[i][0] * pD_degen_dict[i][1]
                pD_count = pD_degen_dict[i][2]

            verbosePrint('\t'*5, 'i', i, 'N_i', N_i, 
                    'pA_count', round(pA_count, 5), 
                    'pD_count', round(pD_count, 5))
            if pD_count != 0 and pA_count != 0: 
                ppD_i = pD_i / float(pD_i + pA_i)
                ppA_i = pA_i / float(pD_i + pA_i)
                N_i_eff = ppD_i * ppA_i / float(0.25) * N_i
                Nc_eff += N_i_eff
                pbias = ppD_i * ppA_i
                for x in range(0, N_i):
                    pbias_list.append(pbias)
                verbosePrint('\t'*6, 'i', i, 'N_i', N_i, 
                        'ppA_i', round(ppA_i, 5), 
                        'ppD_i', round(ppD_i, 5), 
                        'N_i_eff', round(N_i_eff, 5), 
                        'pbias_i', round(pbias, 5))

                ppD_ppA_dict[i] = [ppD_i, ppA_i]
            else:
                for x in range(0, N_i):
                    pbias_list.append(pbias)
                if pD_count == 0 and pA_count != 0:
                    ppD_ppA_dict[i] = [0, 1]
                if pD_count != 0 and pA_count == 0:
                    ppD_ppA_dict[i] = [1, 0]
                if pD_count == 0 and pA_count == 0:
                    ppD_ppA_dict[i] = [0, 0]



        verbosePrint('\t'*5, 'Nc_eff', round(Nc_eff, 5))

        pbias_ave = 0
        if len(pbias_list) != 0:
            pbias_ave = sum(pbias_list) / float(len(pbias_list))

        verbosePrint('\t'*5, 'pbias_ave', round(pbias_ave, 5))
        if Nc_eff != 0:
            S = np.log((Nc_eff) ** 
                    (3 / float(2)) * 
                    np.pi ** 0.5 * pbias_ave / 2)
        verbosePrint('\t'*5, 'S3 (test2)', round(S, 5))

        if S < 0:
            S = 0



    return S, pbias_ave, Nc_eff, ppD_ppA_dict







def forceUnitConversion(forceUnits):
    '''
    Amber = kcal / mol/ Ang / m^2
    Gromacs = kJ / mol / nm / m^2
    '''

    ### for amber
    force_half = 0.5 ** 2
    cal_J = 4184 ** 2
    pmol_pmolec = 6.02214086e23 ** 2
    A_m = 1e-10 ** 2
    A = 1e-10
    gmol_kg = 6.02214086e26

    #'''
    if forceUnits != 'kJ':
        ###top = force_half*cal_J*gmol_kg
        top = cal_J * gmol_kg
        bottom = pmol_pmolec * A_m
        constant = float(top) / float(bottom)
    #'''

    #'''
    if forceUnits == 'kJ':
        ### for gromacs
        nm_m = 1e-9**2
        nm = 1e-9
        kJ_J = 1000 ** 2 ## sqrd as force is sqred
        top = kJ_J * gmol_kg
        bottom = pmol_pmolec * nm_m
        constant = float(top) / float(bottom)
        #constant = 1000
    #'''

    return constant






def SvibCalculations(totFrames, Aclass, temperature, forceUnits, verbosePrint):
    '''
    Get rotated forces^2 and torques^2 matrices from allMoleculeList and
    group into types.
    Then for each molecule type (Nc and extended RAD shell), calc S_vib.

    Units:
    force: Lammps = Kcal/mole-Angstrom, Amber = kcal/mol-Angstrom, 
        SI = kg.m/s^2 = N
    energy: Lammps = Kcal/mole, Amber = Kcal/mole, SI = kg.m^2/s^2 = J(/mol)
    covert Kcal/mole to KJ/mole by * 4.184

    cal to mol = * 4.184

    240.663461 = (1000/6.022E+23)/(1.38E-23*0.5) to get T from KE (kJ/mol)
    '''


    constant = forceUnitConversion(forceUnits)

    for nearest, key2 in Aclass.RADshell_dict.items():
        verbosePrint('\nnearest', nearest)
        for assigned, key3 in key2.items():
            verbosePrint('\tassigned', assigned)

            for RADshell_num, values in key3.items():
                RADshell_num = float(RADshell_num)
                verbosePrint('\n\t\tRADshell_num', RADshell_num, 
                        'count', values[5])
                forces = values[0]
                torques = values[1]
                pe = values[2]
                ke = values[3]
                Temp = values[4]
                count = values[5]
                weight = values[6]
                weightNorm = weight / float(totFrames)
                countNorm = count / float(totFrames)
                zpe = 0


                Strans_qm, Srot_qm, pe_mean, ke_mean, zpe, T = \
                        0, 0, 0, 0, 0, None

                ###setting temp here
                T = temperature 

                try:
                    force_sqrd_SI = np.multiply(forces, constant)
                    Strans_qm, zpe_trans, eigenvalues = Svib_calc(
                            force_sqrd_SI, T, 
                            cov=False, out=False, outputEigenValues=True)
                    zpe = zpe_trans
                    verbosePrint(
                            '\n\t\t\tave forces:\n\t\t\t%s'\
                                    '\n\t\t\t%s\n\t\t\t%s' 
                            % (force_sqrd_SI[0], force_sqrd_SI[1], 
                                force_sqrd_SI[2]), 
                            '\n\t\t\tStrans', Strans_qm)

                    verbosePrint('\n\t\t\teigenvalues:\n\t\t\t%s' % 
                            (eigenvalues))
                        

                    torque_sqrd_SI = np.multiply(torques, constant)
                    Srot_qm, zpe_rot, eigenvalues = Svib_calc(
                            torque_sqrd_SI, T, 
                            cov=False, out=False, outputEigenValues=True)
                    zpe += zpe_rot
                    verbosePrint(
                            '\n\t\t\tave torques:\n\t\t\t%s'\
                                    '\n\t\t\t%s\n\t\t\t%s'
                            % (torque_sqrd_SI[0], torque_sqrd_SI[1], 
                                torque_sqrd_SI[2]),
                            '\n\t\t\tSrot', Srot_qm)

                    verbosePrint('\n\t\t\teigenvalues:\n\t\t\t%s' % 
                            (eigenvalues))


                except (TypeError, ValueError): #force is a zero
                    pass


                pe = pe * 4.184 #kcal to kJ
                verbosePrint('\n\t\t\tave pe: ', pe)
                ke = ke * 4.184
                verbosePrint('\n\t\t\tave ke: ', ke)


                if Strans_qm != None:
                    Strans_qm = Strans_qm * 8.314
                if Strans_qm == None:
                    Strans_qm = 0
                if Srot_qm != None:
                    Srot_qm = Srot_qm * 8.314
                if Srot_qm == None:
                    Srot_qm = 0


                Aclass.allVariables_dict[nearest][assigned]\
                        [RADshell_num]['Strans'] = \
                        Strans_qm, count
                Aclass.allVariables_dict[nearest][assigned]\
                        [RADshell_num]['Srot'] = Srot_qm, count
                if pe != 0 and ke != 0:
                    Aclass.allVariables_dict[nearest][assigned]\
                            [RADshell_num]['PE'] = pe, count
                    Aclass.allVariables_dict[nearest][assigned]\
                            [RADshell_num]['KE'] = ke, count
                Aclass.allVariables_dict[nearest][assigned]\
                        [RADshell_num]['count'] = \
                        countNorm, count





def Svib_calc(matrix_SI, T, cov, out, *args, **kwargs):
    '''
    Calculate Svib from matrix with SI units.
    '''

    outputEigenValues = kwargs.get('outputEigenValues', False)
    eigenvalues = kwargs.get('eigenvalues', False)

    kB = 1.38064852e-23
    #if T == None:
    #T = 298 #over-write inputted T for now
    T = float(T) ##uses input T
    h = 1.0545718e-34 # value of hbar
    #h = 6.62607004e-34
    c = 30000000000
    NA = 6.022e+23
    #print(T)


    #verbosePrint(matrix_SI)

    if eigenvalues == False:
        if cov == True:
            eigenvalues, eigenvectors = LA.eig(matrix_SI)
        if cov == False:
            eigenvalues = matrix_SI.diagonal() #do this for diagonal only
                #CHECK

    
    if out == True:
        if cov == True:
            print('\n\t\t\teigenvectors:\n\t\t\t%s\n\t\t\t%s\n\t\t\t%s' 
                    % (eigenvectors[0], eigenvectors[1], 
                        eigenvectors[2]), 
                    '\n\t\t\teigenvalues:\n\t\t\t%s' % (eigenvalues))
        if cov == False:
            print('\n\t\t\teigenvalues:\n\t\t\t%s' % (eigenvalues))


    frequencies = []
    fraction = []
    sums = []
    ZPE = []
    for value in eigenvalues:
        w = value/(kB*T)
        w = w ** 0.5
        frequencies.append(w)
        zpe = 0.5 * w * h * NA
        ZPE.append(zpe)

    for freq in frequencies:
        #this is for qm
        interim = (h*freq)/(kB*T)
        fraction.append(interim)

    for value in fraction:
        #verbosePrint(value)
        if value != 0:
            try:
                mid = (math.exp(value)-1)
                mid2 = (math.log(1-math.exp(-value)))
                mid3 = value/mid-mid2
                sums.append(mid3)
            except (ValueError, TypeError, OverflowError):
                sums.append(0)
        else:
            sums.append(0)

    S_qm = sum(sums)
    ZPE_qm = sum(ZPE)
    #Svib_qm = NA * kB * S_qm
    Svib_qm = S_qm


    if outputEigenValues == True:
        return Svib_qm, ZPE_qm, eigenvalues
    else:
        return Svib_qm, ZPE_qm
    




def WMSvibCalculations(Aclass, totFrames, temperature, 
        forceUnits, verbosePrint):
    '''
    Get forces and torques for molecule and UA levels with
    shells as x
    '''
   

    for nearest, resname_Ndih_key in \
            sorted(list(Aclass.WM_FT_shell_dict.items())):
        if 'Any' in nearest:
            verbosePrint('\n\n\n***** NEAREST: %s *****' % (nearest))
            for resname_Ndih, shellNum_key in \
                    sorted(list(resname_Ndih_key.items())):
                resname = resname_Ndih[0]
                for RADshellDist, values in \
                        sorted(list(shellNum_key.items())):
                    verbosePrint('\n\n\n***** SHELL_NUM or DIST: %s *****' % 
                            (RADshellDist))
                    RADshellDist = float(RADshellDist)
                    count = values[7]
                    #countNorm = count / float(totFrames)
                    Svib_qm, Strans_qm, Srot_qm, Strans_qm_UA, \
                            Srot_qm_UA, T_mean = process_FT(
                                    Aclass, nearest, resname_Ndih, 
                                    values, temperature, 
                                    forceUnits, verbosePrint)

                    Aclass.allVariables_dict[nearest][resname]\
                            [RADshellDist]['WM_Strans'] = \
                            Strans_qm * 8.314, count
                    Aclass.allVariables_dict[nearest][resname]\
                            [RADshellDist]['WM_Srot'] = \
                            Srot_qm * 8.314, count
                    Aclass.allVariables_dict[nearest][resname]\
                            [RADshellDist]['WM_UA_Strans'] = \
                            Strans_qm_UA * 8.314, count
                    Aclass.allVariables_dict[nearest][resname]\
                            [RADshellDist]['WM_UA_Srot'] = \
                            Srot_qm_UA * 8.314, count



def process_FT(Aclass, nearest, resname_Ndih, values, 
        temperature, forceUnits, verbosePrint):
    '''
    Get forces and torques for molecule and UA levels  - global
    '''
    

    WM_eigenvalues = []
    WM_UA_eigenvalues = []

    resname = resname_Ndih[0][0]
    Ndih = resname_Ndih[1]
    forces = values[0]
    torques = values[1]
    pe = values[2]
    ke = values[3]
    T = values[4]
    UAs_forces = values[5]
    UAs_torques = values[6]
    count = values[7]

    verbosePrint('\n\n\t***** MOLECULE LEVEL: %s *****' % (resname), 
            'count:', count)
    verbosePrint('\t\t--WM', resname)

    Strans_qm, Srot_qm, pe_mean, ke_mean, zpe, T_mean, \
            eigenvalues = FT_S(forces, torques, pe, 
            ke, forceUnits, temperature, verbosePrint, print_out=False)
    verbosePrint('\n\t\t\tTmean', T_mean, '\n')

    for ev in eigenvalues:
        e = ev + [resname]
        WM_eigenvalues.append(e)


    if type(UAs_forces) != type(int) and type(UAs_torques) != type(int):
        verbosePrint('\t\t--WM_UAs', resname)
        Strans_qm, Srot_qm, pe_mean, ke_mean, zpe, T_mean, \
                eigenvalues = FT_S(UAs_forces, UAs_torques, 
                pe, ke, forceUnits, temperature, verbosePrint, print_out=False)
        verbosePrint('\n\t\t\tTmean', T_mean, '\n')

        for ev in eigenvalues:
            e = ev + [resname]
            WM_UA_eigenvalues.append(e)


    #if T_mean != None:
        #temperature = T_mean

    verbosePrint('\t'*2, '--WM eigenvalues')
    WM_force_eigenvalues = []
    WM_torque_eigenvalues = []
    WM_eigenvalues.sort(key=lambda x: x[0])
    for ev in WM_eigenvalues:
        verbosePrint('\t'*3, ev)
        if ev[1] == 'force':
            WM_force_eigenvalues.append(ev[0])
        elif ev[1] == 'torque':
            WM_torque_eigenvalues.append(ev[0])
        else:
            continue


    WM_eigenvalues1 = sorted(list(map(lambda x: x[0], WM_eigenvalues)))
    Svib_qm, zpe_trans = Svib_calc(None, 
            temperature, cov=True, out=False, 
            eigenvalues=WM_eigenvalues1)
    verbosePrint('\n\t\t\tSvib WM', Svib_qm)

    Strans_qm, zpe_trans = Svib_calc(None, temperature, cov=True, 
            out=False, eigenvalues=WM_force_eigenvalues)
    verbosePrint('\n\t\t\tStrans WM', Strans_qm)

    Srot_qm, zpe_trans = Svib_calc(None, temperature, cov=True, 
            out=False, eigenvalues=WM_torque_eigenvalues)
    verbosePrint('\n\t\t\tSrot WM', Srot_qm)

    verbosePrint()


    verbosePrint('\t'*2, '--WM_UAs eigenvalues')
    WM_UA_eigenvalues.sort(key=lambda x: x[0])
    for ev in WM_UA_eigenvalues:
        verbosePrint('\t'*3, ev)



    verbosePrint()
    verbosePrint('\t***** WM_UA LEVEL: %s *****' % (resname))

    force_eigenvalues = []
    torque_eigenvalues = []
    for ev in WM_UA_eigenvalues:
        if ev[1] == 'force':
            force_eigenvalues.append(ev[0])
        elif ev[1] == 'torque':
            torque_eigenvalues.append(ev[0])
        else:
            continue

    torque_eigenvalues = sorted(torque_eigenvalues)
    force_eigenvalues = sorted(force_eigenvalues)

    verbosePrint('\n\t\t--All WM_UA torque eigenvalues:')
    for ev in torque_eigenvalues:
        verbosePrint('\t'*3, ev)

    verbosePrint('\n\t\t--All WM_UA force eigenvalues:')
    for ev in force_eigenvalues:
        verbosePrint('\t'*3, ev)


    force_eigenvalues = force_eigenvalues[6:] 
            # remove 6 smallest eigenvals
            #these correspond to WM level evs
    if Ndih != None: #halve Ndih largest remaining eigenvals
        verbosePrint('\n\t\tNdih: %s' % (Ndih))
        N_ev = np.array(force_eigenvalues[0:Ndih]) 
                #lowest Ndih evs are 1/2
        N_ev_halved = []
        for ev_type in N_ev:
            ev_half = float(ev_type) / float(4)
            N_ev_halved.append(ev_half)

        force_eigenvalues = N_ev_halved + force_eigenvalues[Ndih:]


    verbosePrint('\n\t\t--(-6) WM_UA torque eigenvalues:')
    for ev in torque_eigenvalues:
        verbosePrint('\t'*3, ev)

    verbosePrint('\n\t\t--(-6) WM_UA forces eigenvalues')
    for ev in force_eigenvalues:
        verbosePrint('\t'*3, ev)


    #WM_UA_eigenvalues = sorted(list(map(lambda x: x[0], 
            #WM_UA_eigenvalues)))
    Strans_qm_UA, zpe_trans_UA = Svib_calc(None, 
            temperature, cov=True, out=False,
            eigenvalues=force_eigenvalues)

    verbosePrint('\n\t\t\tStrans WM_UA', Strans_qm_UA)

    Srot_qm_UA, zpe_trans_UA = Svib_calc(None, 
            temperature, cov=True, out=False, 
            eigenvalues=torque_eigenvalues)

    verbosePrint('\n\t\t\tSrot WM_UA', Srot_qm_UA)

    #'''


    return Svib_qm, Strans_qm, Srot_qm, Strans_qm_UA, Srot_qm_UA, T_mean




def FT_S(forces, torques, pe, ke, forceUnits, temperature, 
        verbosePrint, print_out):
    '''
    '''

    constant = forceUnitConversion(forceUnits)
    forces = np.array(forces)
    torques = np.array(torques)
    pe = np.array(pe)
    ke = np.array(ke)

    Strans_qm, Srot_qm, pe_mean, ke_mean, zpe, T_mean = \
            None, None, None, None, 0, None

    ###setting temp here
    T_mean = temperature

    DOF = 0
    if type(forces) != type(int):
        for f in forces[0]:
            if f != 0:
                DOF += 1
            else:
                continue

    if type(torques) != type(int):
        for t in torques[0]:
            if t != 0:
                DOF += 1
            else:
                continue

    eigenvalues_list = []

    if type(forces) != type(int):
        force_sqrd_SI = np.multiply(forces, constant)
        Strans_qm, zpe_trans, eigenvalues = Svib_calc(force_sqrd_SI, 
                T_mean, cov=True, out=False, outputEigenValues=True)
        zpe = zpe_trans
        if print_out == True:
            verbosePrint('\n\t\t\tave forces:')
            for FF in force_sqrd_SI:
                verbosePrint('\n\t\t\t%s' % (FF))

        verbosePrint('\n\t\t\tStrans', Strans_qm)

        for ev in eigenvalues:
            eigenvalues_list.append([ev, 'force'])


    if type(torques) != type(int):
        torque_sqrd_SI = np.multiply(torques, constant)
        Srot_qm, zpe_rot, eigenvalues = Svib_calc(torque_sqrd_SI, 
                T_mean, cov=True, out=False, outputEigenValues=True)
        zpe += zpe_rot
        if print_out == True:
            verbosePrint('\n\t\t\tave torques:')
            for TT in torque_sqrd_SI:
                verbosePrint('\n\t\t\t%s' % (TT))
        verbosePrint('\n\t\t\tSrot', Srot_qm)

        for ev in eigenvalues:
            eigenvalues_list.append([ev, 'torque'])



    pe_mean = np.mean(pe) * 4.184
    verbosePrint('\n\t\t\tave pe: ', pe_mean)

    ke_mean = np.mean(ke) * 4.184
    verbosePrint('\n\t\t\tave ke: ', ke_mean)
    if DOF != 0:
        T_mean = float(ke_mean) / float(DOF) * 240.663461
    else:
        T_mean = float(ke_mean) * 240.663461


    return Strans_qm, Srot_qm, pe_mean, ke_mean, zpe, \
            T_mean, eigenvalues_list






def process_dihedrals(Aclass, verbosePrint):
    '''
    '''

    if len(Aclass.adaptive_dih_dict) != 0:

        ###adaptive dihedrals
        verbosePrint('\nAdaptive Dihedrals p ln p\n')
        for nearest, assigned_key in \
                sorted(list(Aclass.adaptive_dih_dict.items())):
            for assigned, shell_num_key in \
                    sorted(list(assigned_key.items())):
                #assigned_count = 0
                for shell_num, dih_atoms_key in \
                        sorted(list(shell_num_key.items())):
                    verbosePrint(nearest, assigned, shell_num)
                    sum_S = 0
                    shell_count = 0
                    dih_count = 0
                    for dih_atoms, phi_key in \
                            sorted(list(dih_atoms_key.items())):
                        verbosePrint(dih_atoms)
                        dih_count += 1
                        phi_count_list = []
                        max_list = []
                        for phi, count in sorted(list(phi_key.items())):
                            phi_count_list.append([phi, count])
                            shell_count += (count / 
                                    float(len(dih_atoms_key.keys())))
                        N = len(phi_count_list)-1
                        #verbosePrint(phi_count_list)
                        same_dict = nested_dict()
                        for x in range(0, N+1):
                            if phi_count_list[x][1] != 0:
                                if x == 0:
                                    if phi_count_list[x+1][1] <= \
                                            phi_count_list[x][1] and \
                                            phi_count_list[N][1] <= \
                                            phi_count_list[x][1]:
                                        max_list.append(phi_count_list[x])
                                    else:
                                        continue
                                elif x == N:
                                    if phi_count_list[x-1][1] <= \
                                            phi_count_list[x][1] and \
                                            phi_count_list[0][1] <= \
                                            phi_count_list[x][1]:
                                        max_list.append(phi_count_list[x])
                                    else:
                                        continue
                                else:
                                    if phi_count_list[x-1][1] <= \
                                            phi_count_list[x][1] and \
                                            phi_count_list[x+1][1] <= \
                                            phi_count_list[x][1]:
                                        max_list.append(phi_count_list[x])
                                    else:
                                        continue
                            else:
                                continue


                        refined_max_list = []
                        excluded = []

                        for max_count in max_list:
                            for max_count2 in max_list:
                                if max_count != max_count2 and \
                                        len(max_list) > 1 and \
                                        max_count not in refined_max_list:
                                    diff = max_count[0] - max_count2[0]

                                    normDeg = diff % 360
                                    diff2 = min(360-normDeg, normDeg)
                                    #if less than 30 degrees difference 
                                    #then only pick the largest degree to be 
                                    #binned the other gets excluded
                                    if diff2 < 60:
                                        if max_count[1] == max_count2[1]:
                                            if max_count[0] > max_count2[0] \
                                                    and \
                                                    max_count not in \
                                                    refined_max_list:
                                                refined_max_list.append(
                                                        max_count)
                                                excluded.append(max_count2)
                                            else:
                                                continue
                                        elif max_count[1] > max_count2[1]:
                                            if max_count not in \
                                                    refined_max_list:
                                                refined_max_list.append(
                                                        max_count)
                                            else:
                                                continue
                                        else:
                                            continue
                                elif max_count == max_count2 and \
                                        len(max_list) == 1 and \
                                        max_count not in refined_max_list:
                                    refined_max_list.append(max_count)
                                else:
                                    continue

                        for max_count in max_list:
                            for max_count2 in max_list:
                                if max_count != max_count2 and \
                                        len(max_list) > 1 and \
                                        max_count not in refined_max_list:
                                    diff = max_count[0] - max_count2[0]
                                    normDeg = diff % 360
                                    diff2 = min(360-normDeg, normDeg)
                                    if diff2 >= 60 and max_count not in \
                                            refined_max_list and \
                                            max_count not in excluded:
                                        refined_max_list.append(max_count)

                        refined_max_dict = {}
                        for max_count in refined_max_list:
                            refined_max_dict[max_count[0]] = max_count[1]
                        max_phis = list(refined_max_dict.keys())
                        for phi, count in phi_count_list:
                            if phi not in refined_max_dict and count != 0:
                                smallest_diff = 999
                                closest_phi = None
                                for maxPhi in max_phis:
                                    diff = phi - maxPhi
                                    normDeg = diff % 360
                                    diff2 = min(360-normDeg, normDeg)
                                    if diff2 < smallest_diff:
                                        smallest_diff = diff2
                                        closest_phi = maxPhi
                                refined_max_dict[closest_phi] += count

                        ## p ln p calcs finally!
                        sum_count = sum(count 
                                for phi, count in refined_max_dict.items())
                        topo_S = 0
                        for phi, count in refined_max_dict.items():
                            p = count / float(sum_count)
                            S = -p * np.log(p) * 8.314
                            topo_S += S
                            verbosePrint('\t', phi, count, p, S)
                        verbosePrint('\t\t', topo_S)
                        verbosePrint()
                        sum_S += topo_S
                    
                    Aclass.allVariables_dict[nearest][(assigned[0], \
                            assigned[3], assigned[4])]\
                            [float(shell_num)]['conf_AE'] = \
                            sum_S, int(round(shell_count, 0))
                    verbosePrint('\nS_adaptive', sum_S, 'N_dih:', dih_count, 
                            assigned)
                    verbosePrint()






def printAllVariables(num_frames, Aclass, level, name, solvent, 
        waterTuple, verbosePrint):
    '''
    '''

    frames = ''
    if num_frames != None:
        frames = num_frames

    data = open('solventVariables%s%s_%s.csv' % 
            (frames, name, level), 'w')
    data.write(
            '\n'.join(['nearest,assigned,shell_num,variable,value,count']) 
            + '\n')


    data2 = open('soluteVariables%s%s_%s.csv' % 
            (frames, name, level), 'w')
    data2.write('\n'.join(['resName,variable,value,count']) 
            + '\n')


    if level == 'res_atomLevel':
        reduced_dict = nested_dict()
        for nearest, assigned_key in \
                sorted(list(Aclass.allVariables_dict.items())):
            nearest = nearest.split('_')[0]
            for assigned, shellNum_key in sorted(list(assigned_key.items())):
                assigned = assigned[0].split('_')[0]
                if assigned in waterTuple:
                    for shellNum, variable_key in \
                            sorted(list(shellNum_key.items())):
                        tot_count = 0
                        for nearest2, assigned2_key in \
                                Aclass.allVariables_dict.items():
                            for assigned2, shellNum2_key in \
                                    assigned2_key.items():
                                for shellNum2, variable2_key in \
                                        shellNum2_key.items():
                                    for variable2, count in \
                                            variable2_key.items():
                                        nearest3 = nearest2.split('_')[0]
                                        assigned3 = \
                                                assigned2[0].split('_')[0]
                                        if nearest3 == nearest \
                                                and assigned3 == assigned \
                                                and shellNum2 == shellNum \
                                                and variable2 == 'count':
                                            tot_count += count[1]
                                        else:
                                            continue
                        for variable, value in variable_key.items():
                            if variable not in \
                                    reduced_dict[nearest][assigned]\
                                    [shellNum] and value[0] != None:
                                reduced_dict[nearest][assigned][shellNum]\
                                        [variable] = [0, 0]
                            reduced_dict[nearest][assigned][shellNum]\
                                    [variable][0] += \
                                    (value[0] * value[1] / 
                                            float(tot_count))
                            reduced_dict[nearest][assigned][shellNum]\
                                    [variable][1] += \
                                    int(round(value[1], 0))
                else:
                    continue


        for nearest, assigned_key in \
                sorted(list(reduced_dict.items())):
            for assigned, shellNum_key in sorted(list(assigned_key.items())):
                for shellNum, variable_key in \
                        sorted(list(shellNum_key.items())):
                    for variable, value in variable_key.items():
                        if value[0] != None and \
                                assigned in solvent and shellNum == 1:
                            data.write('\n'.join(['%s,%s,%s,%s,%s,%s' % 
                                    (nearest, 
                                    assigned, shellNum, variable, value[0], 
                                    value[1])]) + '\n')
                        else:
                            continue

        ###for solute only
        PE_KE_dict = nested_dict()
        for nearest, assigned_key in \
                sorted(list(Aclass.allVariables_dict.items())):
            for assigned, shellNum_key in sorted(list(assigned_key.items())):
                assigned = list(assigned)
                if '_' in assigned[0]:
                    assigned[0] = assigned[0].split('_')[0]
                for shellNum, variable_key in \
                        sorted(list(shellNum_key.items())): 
                    for variable, value in variable_key.items():
                        if assigned[0] not in solvent and shellNum != 0:
                            if variable in ['PE', 'KE'] and \
                                    value[0] != None:
                                new_var = 'WM_%s' % (variable)
                                if new_var not in PE_KE_dict[nearest]\
                                        [assigned[0]]:
                                    PE_KE_dict[nearest][assigned[0]]\
                                            [new_var] = [0, 0]
                                PE_KE_dict[nearest][assigned[0]]\
                                        [new_var][0] += value[0]
                                PE_KE_dict[nearest][assigned[0]]\
                                        [new_var][1] = value[1]

                            if 'WM_' in variable or variable == 'conf_AE':
                                data2.write('\n'.join(['%s,%s,%s,%s' % 
                                        (assigned[0], 
                                        variable, value[0], 
                                        int(round(value[1], 0)))]) + '\n')
                        else:
                            continue

       
        for nearest, assigned_key in \
                sorted(list(PE_KE_dict.items())):
            for assigned, variable_key in sorted(list(assigned_key.items())):
                for variable, value in sorted(list(variable_key.items())):
                    data2.write('\n'.join(['%s,%s,%s,%s' % 
                            (assigned, 
                            variable, value[0], 
                            value[1])]) + '\n')
        #'''

    else:
        for nearest, assigned_key in \
                sorted(list(Aclass.allVariables_dict.items())):
            for assigned, shellNum_key in sorted(list(assigned_key.items())):
                for shellNum, variable_key in \
                        sorted(list(shellNum_key.items())): 
                    for variable, value in variable_key.items():
                        if assigned[0] in solvent and shellNum == 1 and \
                                value[0] != None:
                            data.write('\n'.join(['%s,%s,%s,%s,%s,%s' % 
                                    (nearest, 
                                    assigned[0], shellNum, 
                                    variable, value[0], 
                                    int(round(value[1], 0)))]) + '\n')
                        else:
                            continue


        #'''
        ###for solute only
        PE_KE_dict = nested_dict()
        for nearest, assigned_key in \
                sorted(list(Aclass.allVariables_dict.items())):
            for assigned, shellNum_key in sorted(list(assigned_key.items())):
                for shellNum, variable_key in \
                        sorted(list(shellNum_key.items())): 
                    for variable, value in variable_key.items():
                        if assigned[0] not in solvent and shellNum != 0:
                            if variable in ['PE', 'KE'] and \
                                    value[0] != None:
                                new_var = 'WM_%s' % (variable)
                                if new_var not in PE_KE_dict[nearest]\
                                        [assigned[0]]:
                                    PE_KE_dict[nearest][assigned[0]]\
                                            [new_var] = [0, 0]
                                PE_KE_dict[nearest][assigned[0]]\
                                        [new_var][0] += value[0]
                                PE_KE_dict[nearest][assigned[0]]\
                                        [new_var][1] = value[1]

                            if 'WM_' in variable or variable == 'conf_AE':
                                data2.write('\n'.join(['%s,%s,%s,%s' % 
                                        (assigned[0], 
                                        variable, value[0], 
                                        int(round(value[1], 0)))]) + '\n')
                        else:
                            continue

        for nearest, assigned_key in \
                sorted(list(PE_KE_dict.items())):
            for assigned, variable_key in sorted(list(assigned_key.items())):
                for variable, value in sorted(list(variable_key.items())):
                    data2.write('\n'.join(['%s,%s,%s,%s' % 
                            (assigned, 
                            variable, value[0], 
                            value[1])]) + '\n')
        #'''


    data.close()
    data2.close()
