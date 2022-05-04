#!/usr/bin/env python

import numpy as np
from itertools import combinations




def contactPopulation(Aclass, RAD_nAtoms, atom_resname, 
        atom_resid, waterTuple, weight):
    '''
    '''

    ####### RESID CONTACT MATRIX POP
    ### contacts defined using QcQn/r^2 between any atoms 
    ### (H or heavy atoms) here!
    if RAD_nAtoms != None:
        #RAD_closestRanked = []
        #nRAD_closestRanked_separated = {}
        for nInfo in RAD_nAtoms:
            nResid = nInfo[1]
            nResname = nInfo[2]
            if nResname not in waterTuple and \
                    atom_resname not in waterTuple and\
                    nResid != atom_resid:
                if (nResname, nResid) not in \
                        Aclass.resid_contact_matrix_dict\
                        [(atom_resname, atom_resid)]:
                    Aclass.resid_contact_matrix_dict\
                            [(atom_resname, atom_resid)]\
                            [(nResname, nResid)] = 0
                    Aclass.resid_contact_matrix_dict\
                            [(nResname, nResid)]\
                            [(atom_resname, atom_resid)] = 0
                Aclass.resid_contact_matrix_dict\
                        [(atom_resname, atom_resid)]\
                        [(nResname, nResid)] += weight #1

            else:
                continue


def contactPopulationUA(Aclass, RAD_nAtoms, atom_name, atom_resname, 
        atom_resid, waterTuple, weight):
    '''
    '''

    ####### RESID CONTACT MATRIX POP
    ### contacts defined using QcQn/r^2 between any atoms 
    ### (H or heavy atoms) here!
    atom_resname = atom_resname.split('_')[0]
    if RAD_nAtoms != None:
        #RAD_closestRanked = []
        #nRAD_closestRanked_separated = {}
        for nInfo in RAD_nAtoms:
            nAtom = nInfo[0]
            nResid = nInfo[1]
            nResname = nInfo[2]
            if nResname not in waterTuple and \
                    atom_resname not in waterTuple and\
                    nResid != atom_resid:
                if (nAtom, nResname, nResid) not in \
                        Aclass.resid_contact_matrix_dict\
                        [(atom_name, atom_resname, atom_resid)]:
                    Aclass.resid_contact_matrix_dict\
                            [(atom_name, atom_resname, atom_resid)]\
                            [(nAtom, nResname, nResid)] = 0
                    Aclass.resid_contact_matrix_dict\
                            [(nAtom, nResname, nResid)]\
                            [(atom_name, atom_resname, atom_resid)] = 0
                Aclass.resid_contact_matrix_dict\
                        [(atom_name, atom_resname, atom_resid)]\
                        [(nAtom, nResname, nResid)] += weight #1

            else:
                continue




def SorPopulation(Aclass, atom_name, atom_resname, 
        nearest_resname, RAD_str, RADshell_num, 
        RADshell_num_dist, waterTuple, N_H, 
        As, Ds, weight):


    Ds = tuple(Ds)
    As = tuple(As)
    DA_shell = Ds + As
    DA_shell = tuple(sorted(DA_shell))
    #print (DA_shell)


    for RR in \
            [[RADshell_num_dist, str(RADshell_num)], ['0', '0']]:

        RADshell_num_dist2 = RR[0]
        RADshell_num2 = RR[1]

        #### As and Ds as they are for pijs
        for ADs in [['A', As], ['D', Ds]]:
            if ADs[1] not in \
                    Aclass.RADshell_Sor_dict[nearest_resname]\
                    [(atom_resname, N_H, atom_name)]\
                    [(RADshell_num_dist2, str(RADshell_num2))]\
                    [RAD_str][ADs[0]]:

                Aclass.RADshell_Sor_dict[nearest_resname]\
                        [(atom_resname, N_H, atom_name)]\
                        [(RADshell_num_dist2, str(RADshell_num2))]\
                        [RAD_str]\
                        [ADs[0]][ADs[1]] = 0

            Aclass.RADshell_Sor_dict[nearest_resname]\
                    [(atom_resname, N_H, atom_name)]\
                    [(RADshell_num_dist2, str(RADshell_num2))]\
                    [RAD_str][ADs[0]][ADs[1]] += weight #1





def runningWeightedAverageFT(weight, weight_stored, count_stored, 
        forces_stored, torques_stored, T_stored, weighted_add, count_add, 
        forces_add, torques_add, T_add, 
        force, torque, KE):
    '''
    Get the running average of force and torque matrices
    '''


    try:
        if any(elem is None for elem in force) == False and \
                any(elem is None for elem in torque) == False:
            try:
                if torques_stored == 0 and forces_stored == 0:
                    weight_add = weight_stored + weight
                    count_add = count_stored + 1
                    forces_add = np.array(force)
                    torques_add = np.array(torque)
            except ValueError:
                weight_add = weight_stored + weight
                count_add = count_stored + 1
                #rolling_weighted_ave = (V_stored * w_stored + V * w) / 
                        #(w_stored + w)
                forces_add = ((forces_stored * weight_stored + 
                        force * weight) / float(weight_stored + weight))
                torques_add = ((torques_stored * weight_stored + 
                        torque * weight) / float(weight_stored + weight))

                if KE != None:
                    try:
                        DOF, T = 0, 0
                        for f in force[0]:
                            if f != 0:
                                DOF += 1
                            else:
                                continue

                        for t in torque[0]:
                            if t != 0:
                                DOF += 1
                            else:
                                continue

                        if DOF != 0:
                            T = KE / float(DOF) * 240.663461
                        if DOF == 0:
                            T = KE * 240.663461    
                        T_add = ((T + T_stored * count_stored) / 
                                float(count_add))
                    except TypeError: #force is a zero
                        pass

    except TypeError:
        pass

    return weight_add, count_add, forces_add, torques_add, T_add


def SvibPopulation(Aclass, level, atom_name, atom_resname, N_H, atom_num, 
        nearest_resname, RADshell_num_dist, waterTuple, 
        UAweightedForces, UAweightedTorques, PE, KE, MweightedForces, 
        MweightedTorques, Ndih, molecule_UA_F, molecule_UA_T, weight):



        #### DICTS FOR SVIB_ECALC

        ###switch level for solutes only to get WM Svib
        if level == 'res_atomLevel' and \
                atom_resname not in waterTuple:
            level == None
            atom_resname = atom_resname.split('_')[0]
        if level == 'residLevel_resname' and \
                atom_resname not in waterTuple:
            level == None
            nearest_resname = nearest_resname.split('_')[0] 
        atom_info = (atom_resname, N_H, atom_name)

        #print(nearest_resname, atom_resname)
        ###running weighted average of X

        if RADshell_num_dist not in Aclass.RADshell_dict\
                [nearest_resname][atom_info]:
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info]\
                    [RADshell_num_dist] = [0, 0, 0, 0, 0, 0, 0]

        if '0' not in Aclass.RADshell_dict\
                [nearest_resname][atom_info]:
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info]\
                    ['0'] = [0, 0, 0, 0, 0, 0, 0]

        for RADshell_num_dist2 in [RADshell_num_dist, '0']:
            forces_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][0]
            torques_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][1]
            PE_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][2]
            KE_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][3]
            T_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][4]
            count_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][5]
            weight_stored = Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][6]

            forces_add = forces_stored
            torques_add = torques_stored
            T_add = T_stored
            count_add = count_stored
            weight_add = weight_stored

            weight_add, count_add, forces_add, torques_add, T_add = \
                    runningWeightedAverageFT(weight, weight_stored, 
                    count_stored, forces_stored, 
                    torques_stored, T_stored, weight_add, count_add, 
                    forces_add, torques_add, T_add, 
                    UAweightedForces, UAweightedTorques, KE)

            PE_add = PE_stored
            KE_add = KE_stored

            if count_add != count_stored and PE != None:
                #PE_add = (PE + PE_stored * count_stored) / float(count_add)
                #KE_add = (KE + KE_stored * count_stored) / float(count_add)

                PE_add = ((PE_stored * weight_stored + 
                        PE * weight) / float(weight_stored + weight))
                KE_add = ((KE_stored * weight_stored + 
                        KE * weight) / float(weight_stored + weight))


            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][0] = forces_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][1] = torques_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][2] = PE_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][3] = KE_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][4] = T_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][5] = count_add
            Aclass.RADshell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist2][6] = weight_add


        atom_info = (atom_info, Ndih)

        ##### PROX SHELL FT ASSIGNMENT

        if type(MweightedForces) != type(None):# and \
                #atom_resname in waterTuple: #water only for now

            if RADshell_num_dist not in Aclass.WM_FT_shell_dict\
                    [nearest_resname]\
                    [atom_info]:
                Aclass.WM_FT_shell_dict[nearest_resname]\
                        [atom_info]\
                        [RADshell_num_dist] = \
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]

            forces_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][0]
            torques_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][1]
            PE_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][2]
            KE_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][3]
            T_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][4]
            UAforces_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][5]
            UAtorques_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][6]
            count_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][7]
            weight_stored = Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][8]

            forces_add = forces_stored
            torques_add = torques_stored
            T_add = T_stored
            UAforces_add = UAforces_stored
            UAtorques_add = UAtorques_stored
            count_add = count_stored
            weight_add = weight_stored

            weight_add, count_add, forces_add, torques_add, T_add = \
                    runningWeightedAverageFT(weight, weight_stored, 
                    count_stored, forces_stored, 
                    torques_stored, T_stored, weight_add, count_add, 
                    forces_add, torques_add, T_add, 
                    MweightedForces, MweightedTorques, False)


            if len(molecule_UA_F) != 0 and len(molecule_UA_T) != 0:
                weight_add, count_ignore, UAforces_add, UAtorques_add, T_add = \
                        runningWeightedAverageFT(weight, weight_stored, 
                        count_stored, 
                        UAforces_stored, UAtorques_stored, 
                        T_stored, weight_add, count_add, 
                        UAforces_add, UAtorques_add, T_add, 
                        molecule_UA_F, molecule_UA_T, KE)

            PE_add = PE_stored
            KE_add = KE_stored

            if count_add != count_stored and PE != None:
                #PE_add = (PE + PE_stored * count_stored) / float(count_add)
                #KE_add = (KE + KE_stored * count_stored) / float(count_add)

                PE_add = ((PE_stored * weight_stored + 
                        PE * weight) / float(weight_stored + weight))
                KE_add = ((KE_stored * weight_stored + 
                        KE * weight) / float(weight_stored + weight))


            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][0] = forces_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][1] = torques_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][2] = PE_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][3] = KE_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][4] = T_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][5] = UAforces_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][6] = UAtorques_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][7] = count_add
            Aclass.WM_FT_shell_dict[nearest_resname]\
                    [atom_info][RADshell_num_dist][8] = weight_add




def SconfPopulation(Aclass, nearest_resname, atom_name, atom_resname, N_H, 
        dihedral_phi_list, molecule_size, RADshell_num_dist, weight):
    '''
    '''

    def roundPartial(value, resolution):
        '''
        used for bin widths of 30 degrees
        '''
        return round(value / resolution) * resolution


    if nearest_resname != None and dihedral_phi_list != None:

        assigned = (atom_resname, len(dihedral_phi_list), 
                molecule_size, N_H, atom_name)

        counter = 0
        for dihedral_phi_type in dihedral_phi_list:
            #print (dihedral_phi_type)
            counter += 1
            #print ('c1:', counter)
            dihedral_atoms = (tuple(dihedral_phi_type[0]), counter)
                    #atomnum_list, dihedral index
            phi = dihedral_phi_type[1] #angle, degrees
            #dih_index = dihedral_phi_type[2][0] #0, 1, 2
            #dih_type = dihedral_phi_type[2][1] #trans, g-, g+

            ##adaptive method
            rounded_phi = roundPartial(phi, 30)
            #print (phi, rounded_phi)
            all_bins = list(range(-180, 181, 30))
            #print (all_bins)
            for b in all_bins:
                if b not in Aclass.adaptive_dih_dict\
                        [nearest_resname][assigned]\
                        [RADshell_num_dist][dihedral_atoms]:
                    Aclass.adaptive_dih_dict[nearest_resname]\
                            [assigned][RADshell_num_dist]\
                            [dihedral_atoms][b] = 0
            Aclass.adaptive_dih_dict[nearest_resname]\
                    [assigned][RADshell_num_dist]\
                    [dihedral_atoms][rounded_phi] += weight #1



