#!/usr/bin/env python




class atom_info(object):
    '''
    Class for all information attributed to atoms and molecules in the system.
    '''
    def __init__(self, atom_num, atom_name, mass, charge, resid, 
            resname, bonded_to_atom_num, dihedral_list): #order important

        self.atom_num = atom_num
        self.atom_name = atom_name
        self.resid = resid
        self.resname = resname
        self.mass = mass
        self.charge = charge
        self.bonded_to_atom_num = bonded_to_atom_num
        self.dihedral_list = dihedral_list 
            #list of lists of atom nums in dihedrals, not including Hs

        self.molecule_atomNums = None #list of bonded UA atom nums
        self.bonded_to_resids = None #list of resids residue is bonded to

        self.coords = None
        self.forces = None
        self.nearest_sorted_array = None #list of (atom_nums, distances) 
                #from nearest to furthest within a set cutoff distance, 
                #not including Hs
        self.nearest_all_atom_array = None #list containing sorted by 
                #dist (atom_num, dist) for all atoms, including Hs
        self.PKenergy = None
        self.UA_PKenergy = None

        self.bondedUA_H = None #[num_bondedUAs, num_bonded Hs]
        self.nearest_atom = None #HB acceptor all atom info
        self.nearest_Hs = [] #HB
        self.dist = None #HB distance between donor and acceptor
        self.broken = None #HB
        self.RAD = None # list of atoms.all_info
        self.RAD_dist = None # list of (atoms.all_info, dist from ref)

        self.UAweightedForces = None
        self.UAweightedTorques = None
        self.MweightedForces = None
        self.MweightedTorques = None
        self.molecule_UA_Fs = None
        self.molecule_UA_Ts = None
        self.WMprincipalAxis = None

        self.dihedral_phi_type = None #unique dihedral angles, in degrees
        self.nearestAnyNonlike = None #tuple (atom.all_info, dist)
        self.RAD_shell_ranked = None #list of 3 tuple
                #RAD shell ranked, acceptors ranked, donors ranked by RAD
        self.RAD_nAtoms = None
        self.hydrationShellRAD_solute = None #nearest solute molecule centric



    def __repr__(self): #return the output below if whole object is printed
        return "%s %s %s %s %s %s %s" % (self.atom_num, self.atom_name, 
                self.mass, self.charge, self.resid, self.resname, 
                self.bonded_to_atom_num)





def clearClass(all_data):
    '''
    For variables in the class, clear after each frame is analysed.
    '''
    
    def reset(self):
        '''
        reset attributes in a class. Except those from topology.
        This prevents memory overload from each frame.
        '''
        dic = vars(self)
        noEdit = ['atom_num', 'atom_name', 'resid', 'resname', 'mass', 
                'charge', 'bonded_to_atom_num', 'nearest_Hs', 
                'molecule_atomNums', 'bonded_to_resids', 
                'dihedral_list', 'bondedUA_H']
        for i in dic.keys():
            if i not in noEdit:
                dic[i] = None
            else:
                continue
        dic['nearest_Hs'] = []

    # clear all objects from previous frame
    for x in range(0, len(all_data)):
        atom = all_data[x]
        reset(atom)



