#!/usr/bin/env python


from collections import defaultdict

nested_dict = lambda: defaultdict(nested_dict) 
        #create nested dict in one go




class variables_dicts(object):
    '''
    Main class containing dicts for all
    thermodynamic (Energy and Entropy) calculations per proximity
    shell in the system.
    '''

    def __init__(self, initiate):

        self.initiate = None


        #### dict with variables named rather than ordered
        self.allVariables_dict = nested_dict() #*

        ##get info about weight of bias in biased simulations
        self.weighting_list = None
        self.weighting_chunks = None

        ###### For Sor population
        self.RADshell_Sor_dict = nested_dict() #for S_or, pijs
        self.Sor_reference_dict = nested_dict()


        ###### For Svib / E population
        self.RADshell_dict = nested_dict()
        #### FT decomposed to prox or dist
        self.WM_FT_shell_dict = nested_dict()


        ### dicts for dihedrals
        self.initiate = None
        ##bin dihs into 30 degrees bin, uses p ln p (not matrix)
        self.adaptive_dih_dict = nested_dict()


        ### contacts between resids with central molecule resid using RAD
        self.resid_contact_matrix_dict = nested_dict()



