#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:01:32 2023

@author: ckadelka
"""

##Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import canalizing_function_toolbox as can
import load_database as db
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import jensenshannon
import scipy.special
import scipy.stats as stats
import matplotlib
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.size': 9})

models_to_keep = ['T-Cell Signaling 2006_16464248.txt',
                  '27765040_tabular.txt',
                  'ErbB (1-4) Receptor Signaling_23637902.txt',
                  'HCC1954 Breast Cell Line Long-term ErbB Network_24970389.txt',
                  'T-LGL Survival Network 2011_22102804.txt',
                  'Predicting Variabilities in Cardiac Gene_26207376.txt',
                  'Lymphopoiesis Regulatory Network_26408858.txt',
                  'Lac Operon_21563979.txt',
                  'MAPK Cancer Cell Fate Network_24250280.txt',
                  'Septation Initiation Network_26244885.txt',
                  '29632237.txt',
                  '25063553_OR_OR.txt',
                  '19622164_TGF_beta1.txt',
                  '23658556_model_10.txt',
                  '23169817_high_dna_damage.txt',
                  '28426669_ARF10_greater_ARF5.txt',
                  '21450717_model_5_2.txt',
                  'Guard Cell Abscisic Acid Signaling_16968132.txt',
                  'FGF pathway of Drosophila Signaling Pathways_23868318.txt',
                  'Death Receptor Signaling_20221256.txt'
                  ]

models_to_exclude_manually_because_similar_from_same_PID = ['Trichostrongylus retortaeformis_22253585.txt',
                                                            'Bordetella bronchiseptica_22253585.txt']


def load_models_included_in_meta_analysis(max_degree=12,max_N=1000,similarity_threshold=0.9,folders=['update_rules_cell_collective/', 'update_rules_models_in_literature_we_randomly_come_across/'],models_to_keep=[],models_to_exclude_manually_because_similar_from_same_PID=[]):
## load the database, choose low max_n for quick results and to only look at small models
    [Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_not_loaded] = db.load_database(folders,max_degree=max_degree,max_N=max_N)
    Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,similar_sets = db.exclude_similar_models(Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,similarity_threshold=similarity_threshold,USE_JACCARD = False,models_to_keep=models_to_keep,models_to_exclude_manually_because_similar_from_same_PID=models_to_exclude_manually_because_similar_from_same_PID)
    n_variables = np.array(list(map(len,variabless)))
    n_constants = np.array(list(map(len,constantss)))
    return Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,models_not_loaded,similar_sets,n_variables,n_constants,max_degree
    

def compute_type_of_each_regulation_in_all_networks(Fs):
    type_of_each_regulation = []
    for i,F in enumerate(Fs):
        type_of_each_regulation.append(compute_type_of_each_regulation_in_one_network(F))
    return type_of_each_regulation


def compute_type_of_each_regulation_in_one_network(F):
    type_of_each_regulation = []
    for f in F:
        if len(f)==0: #happens if actual degree_f > max_degree
            type_of_each_regulation.append([])
            continue
        (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
        dummy = []
        for el in monotonic:
            dummy.append(el)
        type_of_each_regulation.append(dummy)
    return type_of_each_regulation


def flatten(l):
    return [item for sublist in l for item in sublist]


def generate_threshold_rule_ising_mod(f,AUTOREGULATION_INDEX=None,type_of_each_regulation=None):
    if type_of_each_regulation is None:
        type_of_each_regulation = []
        (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
        for el in monotonic:
            type_of_each_regulation.append(el)
    type_of_each_regulation = np.array(type_of_each_regulation)
    n = len(type_of_each_regulation)
    assert np.all(type_of_each_regulation != 'not monotonic'),'conditional variable detected, cannot generate threshold rule'
    assert np.all(type_of_each_regulation != 'not essential'),'non-essential variable detected, simplify function first'
    AUTOREGULATION_ADDED = False
    left_hand_side = ' + '.join(['(x'+str(i)+' == '+('1)' if type_reg == 'increasing' else '0)') for i,type_reg in zip(range(n),type_of_each_regulation)])
    f_threshold = can.f_from_expression(left_hand_side + ' > '+str(n//2))[0]
    return f_threshold,AUTOREGULATION_ADDED
    

def generate_threshold_rule_01(f,AUTOREGULATION_INDEX=None,type_of_each_regulation=None):
    if type_of_each_regulation is None:
        type_of_each_regulation = []
        (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
        for el in monotonic:
            type_of_each_regulation.append(el)
    type_of_each_regulation = np.array(type_of_each_regulation)
    n = len(type_of_each_regulation)
    assert np.all(type_of_each_regulation != 'not monotonic'),'conditional variable detected, cannot generate threshold rule'
    assert np.all(type_of_each_regulation != 'not essential'),'non-essential variable detected, simplify function first'
    AUTOREGULATION_ADDED = False
    left_hand_side = ''.join([(' + ' if type_reg == 'increasing' else ' - ')+'x'+str(i) for i,type_reg in zip(range(n),type_of_each_regulation)])
    if AUTOREGULATION_INDEX==None:
        f_threshold = can.f_from_expression(left_hand_side + ' > 0')[0]
        f_threshold.extend( can.f_from_expression(left_hand_side + ' >= 0')[0] )
        AUTOREGULATION_ADDED = True
    else:
        #f_threshold = can.f_from_expression(left_hand_side + (' >=' if type_of_each_regulation[AUTOREGULATION_INDEX]=='increasing' else ' >') + ' 0')[0]
        f_threshold = can.f_from_expression(left_hand_side +' > 0 or ( '+left_hand_side+'  == 0 and x'+str(AUTOREGULATION_INDEX)+' == 1)')[0]
    return f_threshold,AUTOREGULATION_ADDED
   

def generate_threshold_rule_01_mod(f,AUTOREGULATION_INDEX=None,type_of_each_regulation=None):
    if type_of_each_regulation is None:
        type_of_each_regulation = []
        (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
        for el in monotonic:
            type_of_each_regulation.append(el)
    type_of_each_regulation = np.array(type_of_each_regulation)
    n = len(type_of_each_regulation)
    assert np.all(type_of_each_regulation != 'not monotonic'),'conditional variable detected, cannot generate threshold rule'
    assert np.all(type_of_each_regulation != 'not essential'),'non-essential variable detected, simplify function first'
    AUTOREGULATION_ADDED = False
    if np.all(type_of_each_regulation == 'increasing'):
        f_threshold = [1]*(2**n)
        f_threshold[0] = 0
    elif np.all(type_of_each_regulation == 'decreasing'):
        f_threshold = [0]*(2**n)
        f_threshold[0] = 1
    else:
        left_hand_side = ''.join([(' + ' if type_reg == 'increasing' else ' - ')+'x'+str(i) for i,type_reg in zip(range(n),type_of_each_regulation)])
        if AUTOREGULATION_INDEX==None:
            f_threshold = can.f_from_expression(left_hand_side + ' > 0')[0]
            f_threshold.extend( can.f_from_expression(left_hand_side + ' >= 0')[0] )
            AUTOREGULATION_ADDED = True
        else:
            f_threshold = can.f_from_expression(left_hand_side +' > 0 or ( '+left_hand_side+'  == 0 and x'+str(AUTOREGULATION_INDEX)+' == 1)')[0]
    return f_threshold,AUTOREGULATION_ADDED


def generate_threshold_rule_ising(f,AUTOREGULATION_INDEX=None,type_of_each_regulation=None):
    if type_of_each_regulation is None:
        type_of_each_regulation = []
        (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
        for el in monotonic:
            type_of_each_regulation.append(el)
    type_of_each_regulation = np.array(type_of_each_regulation)
    n = len(type_of_each_regulation)
    assert np.all(type_of_each_regulation != 'not monotonic'),'conditional variable detected, cannot generate threshold rule'
    assert np.all(type_of_each_regulation != 'not essential'),'non-essential variable detected, simplify function first'
    AUTOREGULATION_ADDED = False
    left_hand_side = ' + '.join(['(x'+str(i)+' == '+('1)' if type_reg == 'increasing' else '0)') for i,type_reg in zip(range(n),type_of_each_regulation)])
    if n%2==0:
        if AUTOREGULATION_INDEX is None:
            f_threshold = can.f_from_expression(left_hand_side + ' > '+str(n//2))[0]
            f_threshold.extend( can.f_from_expression(left_hand_side + ' >= '+str(n//2))[0] )
            AUTOREGULATION_ADDED = True
        else:
            f_threshold = can.f_from_expression(left_hand_side +' > '+str(n//2)+' or ( '+left_hand_side+'  == '+str(n//2)+' and x'+str(AUTOREGULATION_INDEX)+' == 1)')[0]
    else:
        if AUTOREGULATION_INDEX is None:
            f_threshold = can.f_from_expression(left_hand_side + ' > '+str((n)//2))[0]
            AUTOREGULATION_ADDED = False
        else:
            f_threshold = can.f_from_expression(left_hand_side +' > '+str((n)//2))[0]
    return f_threshold,AUTOREGULATION_ADDED
    

def get_threshold_network(F,I,generate_threshold_rule_function = generate_threshold_rule_01):
    F_threshold = []
    I_threshold = []
    for i,(f,regulators) in enumerate(zip(F,I)):
        if len(f)==1:
            F_threshold.append(f)
        elif len(f) == 0: #if degree = len(I) > max_degree
            F_threshold.append(f)
            
        else:
            try:
                AUTOREGULATION_INDEX = list(regulators).index(i)
            except ValueError: #no auto regulation
                AUTOREGULATION_INDEX = None
            type_of_each_regulation = []
            (NONDEGENERATED,monotonic) = can.is_monotonic(f,True)
            for el in monotonic:
                type_of_each_regulation.append(el)
            f_threshold,AUTOREGULATION_ADDED = generate_threshold_rule_function(f,AUTOREGULATION_INDEX,type_of_each_regulation)
            F_threshold.append( np.array(f_threshold) )
        if AUTOREGULATION_ADDED:
            I_threshold.append(np.append(i,regulators))
        else:
            I_threshold.append(regulators.copy())
    return F_threshold,I_threshold


def compute_jsd(p, q):
    # Make sure p and q are numpy arrays
    p = np.array(p)
    q = np.array(q)

    # Normalize the probability distributions to sum to 1
    p = p / p.sum()
    q = q / q.sum()

    # Compute Jensen-Shannon divergence using scipy
    jsd = jensenshannon(p, q, base=2)

    return jsd


def get_similarity_of_attractor_spaces(attractors1,attractors2,basin_sizes1,basin_sizes2):
    all_attractor_states = dict()
    n_attractor_states = 0
    p = []
    q = []
    for attractor,basin_size in zip(attractors1,basin_sizes1):
        size_attractor = len(attractor)
        for state in attractor:
            all_attractor_states.update({state:n_attractor_states})
            n_attractor_states += 1
            p.append(basin_size/size_attractor)
            q.append(0)
    for attractor,basin_size in zip(attractors2,basin_sizes2):
        size_attractor = len(attractor)
        for state in attractor:
            try:
                index = all_attractor_states[state]
                q[index] = basin_size/size_attractor
            except KeyError:
                all_attractor_states.update({state:n_attractor_states})
                index = n_attractor_states
                n_attractor_states += 1
                p.append(0)
                q.append(basin_size/size_attractor)  
    return compute_jsd(p,q),p,q


def get_similarity_functions(f1,f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    l_f1 = len(f1)
    l_f2 = len(f2)
    if l_f1 == 2*l_f2:
        f2 = np.append(f2,f2)
    elif l_f2 == 2*l_f1:
        f1 = np.append(f1,f1)
    elif l_f1 == l_f2:
        pass
    else:
        raise Exception("The degree of the two functions must be equal, or off by 1.")
    return np.mean(f1==f2)


def stratify_all_regulatory_rules(Fs,Is,DIFFERENTIATE_AUTOREGULATION = True):
    res = []
    res_f = []
    res_unsorted = []
    dictX = {'increasing' : 'a', 'decreasing' : 'i'}
    
    for j in range(len(Fs)):
        type_of_each_regulation_in_F = compute_type_of_each_regulation_in_one_network(Fs[j])
        for i,(f,regulators,types_regulation) in enumerate(zip(Fs[j],Is[j],type_of_each_regulation_in_F)):
            if 'not monotonic' not in types_regulation and 'not essential' not in types_regulation:
                dummy = list(map(lambda x: dictX[x],types_regulation))
                if DIFFERENTIATE_AUTOREGULATION:
                    try:
                        index_autoregulation = np.where(regulators==i)[0][0]
                        dummy[index_autoregulation] = dummy[index_autoregulation].capitalize()
                    except IndexError:
                        pass
                res_unsorted.append(''.join(dummy))
                dummy.sort()
                dummy = ''.join(dummy)
                res.append(dummy)
                res_f.append(f)
    res = np.array(res)
    res_unsorted = np.array(res_unsorted)
    res_f = np.array(res_f,dtype=object) 
    len_res = np.array(list(map(len,res)))
    
    #restrict to those functions with 2 to 11 inputs
    res = res[np.bitwise_and(len_res>1,len_res<=11)]
    res_unsorted = res_unsorted[np.bitwise_and(len_res>1,len_res<=11)]
    res_f = res_f[np.bitwise_and(len_res>1,len_res<=11)]
    len_res = len_res[np.bitwise_and(len_res>1,len_res<=11)]
    return res,res_unsorted,res_f,len_res


def compute_similarity_between_threshold_and_bio_rules(res_f,res_unsorted,res,threshold_formalisms = [generate_threshold_rule_ising,generate_threshold_rule_ising_mod,generate_threshold_rule_01,generate_threshold_rule_01_mod],formalisms = ['Ising','Ising'+r'$^{*}$','01',r'01$^{*}$']):
    n_threshold_formalism = len(threshold_formalisms)
    unique_res = np.array(list(set(res)))
    dict_unique_res = dict(zip(unique_res,list(range(len(unique_res)))))
    differences_f_to_threshold = [[] for iii in range(n_threshold_formalism)]
    differences_f_to_threshold_stratified = [[[] for _ in range(len(unique_res))] for iii in range(n_threshold_formalism)]
    mean_difference_threshold_per_identifer = []
    n_difference_threshold_per_identifer = []
    for iii,generate_threshold_rule in enumerate(threshold_formalisms):
        for f,identifier_unsorted,identifier in zip(res_f,res_unsorted,res):
            if 'A' in identifier_unsorted:
                AUTOREGULATION_INDEX = identifier_unsorted.index('A')
            elif 'I' in identifier_unsorted:
                AUTOREGULATION_INDEX = identifier_unsorted.index('I')
            else:
                AUTOREGULATION_INDEX = None
            f_threshold = np.array(generate_threshold_rule(f=f,AUTOREGULATION_INDEX=AUTOREGULATION_INDEX)[0])
            if len(f_threshold) > len(f):
                differences_f_to_threshold[iii].append(np.mean(np.append(f,f)!=f_threshold))
            else:
                differences_f_to_threshold[iii].append(np.mean(np.array(f)!=f_threshold))
            differences_f_to_threshold_stratified[iii][dict_unique_res[identifier]].append( differences_f_to_threshold[iii][-1] )
        differences_f_to_threshold = np.array(differences_f_to_threshold,dtype=object)
        
        mean_difference_threshold_per_identifer.append( np.array(list(map(np.mean,differences_f_to_threshold_stratified[iii])),dtype=object) )
        n_difference_threshold_per_identifer.append( np.array(list(map(len,differences_f_to_threshold_stratified[iii])),dtype=object) )
    return differences_f_to_threshold,differences_f_to_threshold_stratified,mean_difference_threshold_per_identifer,n_difference_threshold_per_identifer


if __name__ == '__main__':
    ##Parameters
    max_degree = 14 # load all rules with max_degree or less regulators
    max_n_inputs_simulated_in_attractor_computation = 32 #maximal number of different source node combinations to use when approximating the state space and attractor space 
    n_IC_per_network_in_attractor_computation = 1000 #number of initial conditions to be used when approximating the state space and attractor space of a BN with fixed source nodes 
    n_steps_timeout_when_computing_attractors = 1000 #maximal number of iterations when computing attractors
    max_N = 100000 #i.e., do not exclude models because they are too large   
    COMPUTE_EXACT_STATE_SPACE = False

    ## Load database
    Fs,Is,degrees,degrees_essential,variabless,constantss,models_loaded,models_excluded,models_not_loaded,similar_sets_of_models,n_variables,n_constants,max_degree = load_models_included_in_meta_analysis(max_degree=max_degree,models_to_keep=models_to_keep,models_to_exclude_manually_because_similar_from_same_PID=models_to_exclude_manually_because_similar_from_same_PID)
    N = len(models_loaded) #number of Boolean models
    
    
    #remove non-essential variables
    Fs_essential = []
    Is_essential = []
    for i in range(N):
        dummy = can.get_essential_network(Fs[i],Is[i])
        Fs_essential.append(dummy[0])
        Is_essential.append(dummy[1])


    #get a nice identified (Pubmed ID + x) for each model, where x is used to differentiate different models from the same publication
    pmids_models_loaded = [int(''.join([entry for entry in el.split('.')[0].split('_') if (entry[0] in '123456789' and len(entry)>3)])) for el in models_loaded]
    pmids_to_print = list(map(str,pmids_models_loaded))
    dummy = pd.Series(pmids_models_loaded).value_counts()
    PMIDS_with_multiple_included_models  = list(dummy.index)[:sum(dummy>1)]
    for i,PMID in enumerate(pmids_models_loaded):
        if PMID in PMIDS_with_multiple_included_models:
            pmids_to_print[i] += ' (%s)' %  models_loaded[i].split('_')[0].replace('HCC1954 Breast Cell Line ','').replace(' of Drosophila Signaling Pathway','').replace(' of Drosophila Signaling Pathways','').replace(' from the Drosophila Signaling Pathway','').replace(' of Drosophila Signalling Pathways','')
    
    
    #exclude models that contain conditional regulators, have degree>max_degree or contain constant nodes after simplification
    type_of_each_regulation = compute_type_of_each_regulation_in_all_networks(Fs_essential)
    good_indices = []
    for i in range(N):
        if len(set(flatten(type_of_each_regulation[i])) - set(['increasing','decreasing'])) == 0:
            if max(degrees_essential[i])<=max_degree and min(degrees_essential[i])>0:
                if len(Fs_essential[i]) <= max_N:
                    good_indices.append(i)
    good_indices = np.array(good_indices)
    

    #get initial conditions for state space simulation, 
    #pick a random set of min(2**n_constants,max_n_inputs_simulated_in_attractor_computation) choices for the constants,
    #and per such condition, generate n_IC_per_network_in_attractor_computation choices for the initial conditions of the variables
    #use the same initial conditions for the different threshold formalisms to increase accuracy of the simulations
    initial_sample_points = []
    for i in good_indices:
        if n_constants[i] == 0:
            choices_for_constants_binary_strings = ['']
        elif 2**n_constants[i] <= max_n_inputs_simulated_in_attractor_computation:
            choices_for_constants_binary_strings = [bin(el)[2:].zfill(n_constants[i]) for el in range(2**n_constants[i])]
        else:
            choices_for_constants_binary_strings = [''.join(np.random.choice(['0','1'],n_constants[i])) for el in range(max_n_inputs_simulated_in_attractor_computation)]
            
                    
        initial_sample_points.append([])
        for binary_string_constants in choices_for_constants_binary_strings:
            initial_sample_points[-1].append([])
            for j in range(n_IC_per_network_in_attractor_computation):
                binary_string_variables = ''.join(np.random.choice(['0','1'],n_variables[i]))
                initial_sample_points[-1][-1].append( int(binary_string_variables + binary_string_constants, 2) )

    
    #approximate dynamics for the biological networks
    attractors = []
    number_attractors = []
    basin_sizes = []
    attractor_dicts = []
    state_spaces = []
    n_timeout = []
    for ii,i in enumerate(good_indices):
        F = Fs[i]
        I = Is[i]
        if max_N <= 20 and COMPUTE_EXACT_STATE_SPACE:
            sizeF = len(F)
            dummy = can.get_attractors_synchronous_exact(F, I)
            attractors.append(dummy[0])
            number_attractors.append(dummy[1])
            basin_sizes.append(dummy[2])
            attractor_dicts.append(dummy[3])
            state_spaces.append(dummy[4])
        else:
            attractors.append([])
            number_attractors.append([])
            basin_sizes.append([])
            attractor_dicts.append([])
            state_spaces.append([])
            n_timeout.append([])
            for j in range(min(2**n_constants[i],max_n_inputs_simulated_in_attractor_computation)):
                dummy = can.get_attractors_synchronous(F, I, nsim = n_IC_per_network_in_attractor_computation, 
                                                 n_steps_timeout=n_steps_timeout_when_computing_attractors, 
                                                 initial_sample_points = initial_sample_points[ii][j],
                                                 INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS=False)
                attractors[-1].append(dummy[0])
                number_attractors[-1].append(dummy[1])
                basin_sizes[-1].append(dummy[2])
                attractor_dicts[-1].append(dummy[3])
                state_spaces[-1].append(dummy[5]) 
                n_timeout[-1].append(dummy[6])
            print(i) 
    
    
    #approximate dynamics for the different threshold-based networks
    all_attractors_threshold = []
    all_number_attractors_threshold = []
    all_basin_sizes_threshold = []
    all_attractor_dicts_threshold = []
    all_state_spaces_threshold = []
    all_n_timeout_threshold = []  
    
    all_jsd_attractors = []
    all_overlap_synchronous_state_space = []
    all_overlap_synchronous_state_space_def2 = []
    func_thresholds = [generate_threshold_rule_ising,generate_threshold_rule_ising_mod,generate_threshold_rule_01,generate_threshold_rule_01_mod]
    for func_threshold in func_thresholds:
        suffix = ''.join(func_threshold.__name__.split('_')[3:])
        
        Fs_threshold = []
        Is_threshold = []
        for i in range(N):
            if i in good_indices:
                F_threshold,I_threshold = get_threshold_network(Fs_essential[i],Is_essential[i],func_threshold)
            else:
                F_threshold,I_threshold = [],[]
            Fs_threshold.append(F_threshold)
            Is_threshold.append(I_threshold)
            

        
        if max_N<= 20 and COMPUTE_EXACT_STATE_SPACE:
            left_side_of_truth_table = np.array(list(map(np.array,list(itertools.product([0, 1], repeat = max_N)))))
        attractors_threshold = []
        number_attractors_threshold = []
        basin_sizes_threshold = []
        attractor_dicts_threshold = []
        state_spaces_threshold = []
        n_timeout_threshold = []
        for ii,i in enumerate(good_indices):
            F = Fs_threshold[i]
            I = Is_threshold[i]
            if max_N <= 20 and COMPUTE_EXACT_STATE_SPACE:
                sizeF = len(F)
                dummy = can.get_attractors_synchronous_exact(F, I,left_side_of_truth_table[:2**sizeF,-sizeF:])
                attractors_threshold.append(dummy[0])
                number_attractors_threshold.append(dummy[1])
                basin_sizes_threshold.append(dummy[2])
                attractor_dicts_threshold.append(dummy[3])
                state_spaces_threshold.append(dummy[4])
            else:
                attractors_threshold.append([])
                number_attractors_threshold.append([])
                basin_sizes_threshold.append([])
                attractor_dicts_threshold.append([])
                state_spaces_threshold.append([])
                n_timeout_threshold.append([])
                for j in range(min(2**n_constants[i],max_n_inputs_simulated_in_attractor_computation)):
                    dummy = can.get_attractors_synchronous(F, I, nsim = n_IC_per_network_in_attractor_computation, 
                                                     n_steps_timeout=n_steps_timeout_when_computing_attractors, 
                                                     initial_sample_points = initial_sample_points[ii][j],
                                                     INITIAL_SAMPLE_POINTS_AS_BINARY_VECTORS=False)
                    attractors_threshold[-1].append(dummy[0])
                    number_attractors_threshold[-1].append(dummy[1])
                    basin_sizes_threshold[-1].append(dummy[2])
                    attractor_dicts_threshold[-1].append(dummy[3])
                    state_spaces_threshold[-1].append(dummy[5]) 
                    n_timeout_threshold[-1].append(dummy[6])
                print(i)
                
                
        #compute similarity of the synchronous state space
        overlap_synchronous_state_space = []
        overlap_synchronous_state_space_def2 = []
        if max_N<= 20 and COMPUTE_EXACT_STATE_SPACE:
            b_for_bin2dec = np.array([2**i for i in range(max_N)])[::-1]
            for i in range(len(good_indices)):
                state_space_bin = np.dot(state_spaces[i],b_for_bin2dec[-state_spaces[i].shape[1]:])
                state_space_threshold_bin = np.dot(state_spaces_threshold[i],b_for_bin2dec[-state_spaces_threshold[i].shape[1]:])
                overlap_synchronous_state_space.append(np.mean(state_space_bin == state_space_threshold_bin))
                overlap_synchronous_state_space_def2.append(np.mean(state_spaces_threshold[i]==state_spaces[i]))
        else:
            for ii,i in enumerate(good_indices):
                state_space_dec = np.array([el for j in range(min(2**n_constants[i],max_n_inputs_simulated_in_attractor_computation)) for el in state_spaces[ii][j]])
                state_space_threshold_bin = np.array([el for j in range(min(2**n_constants[i],max_n_inputs_simulated_in_attractor_computation)) for el in state_spaces_threshold[ii][j]])
                overlap_synchronous_state_space.append( np.mean(state_space_dec == state_space_threshold_bin) )
                overlap_synchronous_state_space_def2.append( (n_variables[i] - np.mean([bin(max(a,b)-min(a,b))[2:].count('1') for a,b in zip(state_space_dec,state_space_threshold_bin)])) / n_variables[i] )
        overlap_synchronous_state_space = np.array(overlap_synchronous_state_space)
        overlap_synchronous_state_space_def2 = np.array(overlap_synchronous_state_space_def2)        
        
        order = np.argsort(overlap_synchronous_state_space_def2)
        
        #compute the JSD
        jsd_attractors = []
        if max_N<= 20 and COMPUTE_EXACT_STATE_SPACE:
            for attractors1,attractors2,basin_sizes1,basin_sizes2 in zip(attractors_threshold,attractors,basin_sizes_threshold,basin_sizes):
                jsd_attractors.append(get_similarity_of_attractor_spaces(attractors1,attractors2,basin_sizes1,basin_sizes2)[0])
        else:
            for ii,i in enumerate(good_indices):
                jsd_attractors.append( np.mean([get_similarity_of_attractor_spaces(a1,a2,b1,b2)[0] for a1,a2,b1,b2 in zip(attractors[ii],attractors_threshold[ii],basin_sizes[ii],basin_sizes_threshold[ii])]) )
        jsd_attractors = np.array(jsd_attractors)
        
        
        #store results
        all_attractors_threshold.append(attractors_threshold)
        all_number_attractors_threshold.append(number_attractors_threshold)
        all_basin_sizes_threshold.append(basin_sizes_threshold)
        all_attractor_dicts_threshold.append(attractor_dicts_threshold)
        all_state_spaces_threshold.append(state_spaces_threshold)
        all_n_timeout_threshold.append(n_timeout_threshold)
        all_jsd_attractors.append(jsd_attractors)
        all_overlap_synchronous_state_space.append(overlap_synchronous_state_space)
        all_overlap_synchronous_state_space_def2.append(overlap_synchronous_state_space_def2)
        

        #Compute the Spearman and Pearson correlation of different metrics across the biological networks
        metrics = [np.array(n_variables)[np.array(good_indices)], np.array(n_constants)[np.array(good_indices)], [np.mean(degrees_essential[i]) for i in good_indices],
                   overlap_synchronous_state_space_def2,overlap_synchronous_state_space,1-jsd_attractors]
        
        n_metrics = len(metrics)
        names = ['number variables','number constants','average degree',
                 'overlap functions','overlap state space','overlap attractors']
        spearman_mat = np.ones((n_metrics,n_metrics))
        for i in range(n_metrics):
            for j in range(n_metrics):
                spearman_mat[i,j] = stats.spearmanr(metrics[i],metrics[j])[0]
                spearman_mat[j,i] = spearman_mat[i,j]
        pearson_mat = np.ones((n_metrics,n_metrics))
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                pearson_mat[i,j] = stats.pearsonr(metrics[i],metrics[j])[0]
                pearson_mat[j,i] = pearson_mat[i,j]
        
        #Plot the Spearman and Pearson correlation of different metrics across the biological networks
        for ii,(data,name) in enumerate(zip([spearman_mat,pearson_mat],['spearman','pearson'])):
            f,ax = plt.subplots(figsize=(3.5,3.5))
            im = ax.imshow(data,origin='upper',vmin=-1,vmax=1,cmap=matplotlib.cm.RdBu)
            ax.set_yticks(range(n_metrics))
            ax.set_yticklabels(names)
            ax.set_xticks(range(n_metrics))
            ax.set_xticklabels(names,rotation=90)
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            
            divider = make_axes_locatable(ax)
            caxax = divider.append_axes("right", size='8%', pad=0.1)    
            cbar = f.colorbar(im,cax=caxax,label=name.capitalize()+' correlation coefficient')
            ax.set_title(suffix)
        
            plt.savefig(name+'_correlation_bio_vs_%s.pdf' % suffix,bbox_inches='tight')
    
    #Plot the empirical CDF of different dynamic similarity measures across the biological networks
    colors = ['blue','orange','green','black'] 
    lss = ['-','--','-.',':']
    formalisms = ['Ising','Ising'+r'$^{*}$','01',r'01$^{*}$']

    for metrics,metrics_label in zip([all_overlap_synchronous_state_space_def2,all_overlap_synchronous_state_space,1-np.array(all_jsd_attractors)],
                                     ['mean function-level agreement','overlap state space','overlap attractor space']):
        f,ax = plt.subplots(figsize=(2.5,2.5))
        for jj,(func_threshold,metric) in enumerate(zip(func_thresholds,metrics)):
            suffix = ''.join(func_threshold.__name__.split('_')[3:])    
            ax.plot(np.sort(metric),label=formalisms[jj],color=colors[jj],ls=lss[jj])
        ax.legend(loc='best',frameon=False,ncol=2)
        ax.set_ylabel(metrics_label)
        ax.set_xlabel('networks')
        plt.savefig('cdf_%s.pdf' % metrics_label.replace(' ','_'),bbox_inches='tight')
        
    f,ax = plt.subplots(1,3,figsize=(7.5,2))
    for ii,(metrics,metrics_label) in enumerate(zip([all_overlap_synchronous_state_space_def2,all_overlap_synchronous_state_space,1-np.array(all_jsd_attractors)],
                                                    ['mean function-level agreement','overlap state space','overlap attractor space'])):
        
        for jj,(func_threshold,metric) in enumerate(zip(func_thresholds,metrics)):
            suffix = ''.join(func_threshold.__name__.split('_')[3:])    
            ax[ii].plot(np.sort(metric),label=formalisms[jj],color=colors[jj],ls=lss[jj])
            print(metrics_label,formalisms[jj],np.mean(np.sort(metric))*100)
        if ii==1:
            ax[ii].legend(loc='center',frameon=False,ncol=4,bbox_to_anchor=[0.5,1.1])
        ax[ii].set_ylabel(metrics_label)
        ax[ii].set_xlabel('biological networks')
    plt.subplots_adjust(wspace=0.35)
    plt.savefig('cdf_all.pdf',bbox_inches='tight')
    
    
    names_y = ['mean function-level agreement','overlap state space','overlap attractor space']   
    names_x = names_y + ['number of regulated nodes','number of source nodes','average network connectivity']    
    names_x = names_y + ['network size','average network connectivity'] 

    spearman_mat = np.ones((3*len(formalisms),len(names_x)))
    spearman_mat_p = np.ones((3*len(formalisms),len(names_x)))
    pearson_mat = np.ones((3*len(formalisms),len(names_x)))
    for ii,formalism in enumerate(formalisms):
        metrics_x = [all_overlap_synchronous_state_space_def2[ii],all_overlap_synchronous_state_space[ii],1-all_jsd_attractors[ii]]
        metrics_y = metrics_x + [np.array(n_variables)[np.array(good_indices)], np.array(n_constants)[np.array(good_indices)], [np.mean(degrees_essential[i]) for i in good_indices]]
        metrics_y = metrics_x + [np.array(n_variables)[np.array(good_indices)], [np.mean(degrees_essential[i]) for i in good_indices]]
        #metrics_y = metrics_x + [np.array(n_variables)[np.array(good_indices)], [np.mean(degrees_essential[i]) for i in good_indices],[np.max(degrees_essential[i]) for i in good_indices]]
    
        for i,data_x in enumerate(metrics_x):
            for j,data_y in enumerate(metrics_y):
                spearman_mat[3*ii+i,j] = stats.spearmanr(data_x,data_y)[0]
                spearman_mat_p[3*ii+i,j] = stats.spearmanr(data_x,data_y)[1]
                
                pearson_mat[3*ii+i,j] = stats.pearsonr(data_x,data_y)[0]

   
    #Plot the Spearman and Pearson correlation of different metrics across the biological networks in one plot for all threshold formalisms
    for ii,(data,name) in enumerate(zip([spearman_mat,pearson_mat],['spearman','pearson'])):
        f,ax = plt.subplots(figsize=(3.5,5))
        im = ax.imshow(data,origin='upper',vmin=-1,vmax=1,cmap=matplotlib.cm.RdBu)
        ax.set_yticks(range(3*len(formalisms)))
        ax.set_yticklabels(names_y*len(formalisms))
        ax.set_xticks(range(len(names_x)))
        ax.set_xticklabels(names_x,rotation=90)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        [x1,x2] = ax.get_xlim()
        
        for jj in range(3):
            ax.plot([-.5,5.5],[3*(jj+1)-.5,3*(jj+1)-.5],'k-',lw=0.5)
        for jj,formalism in enumerate(formalisms):
            ax.text(-8,1+3*jj,formalism,va='center',ha='center',rotation=90)
            ax.plot([-7.5]*2,[-.3+3*jj,-.3+2.6+3*jj],'k-',lw=0.5,clip_on=False)
        ax.set_xlim([x1,x2])
        
        divider = make_axes_locatable(ax)
        caxax = divider.append_axes("right", size='8%', pad=0.1)    
        cbar = f.colorbar(im,cax=caxax,label=name.capitalize()+' correlation across 100 biological networks')
    
        plt.savefig(name+'_correlation_bio_vs_all.pdf',bbox_inches='tight')
    
    formalisms = [r'Ising$^{}$', r'Ising$^{*}$', r'01$^{}$', r'01$^{*}$']
    for ii,(corr_function,name) in enumerate(zip([stats.spearmanr,stats.kendalltau],['spearman','kendall rank'])):
        for jj,(name_metric,metric) in enumerate(zip(names_y,[all_overlap_synchronous_state_space_def2,all_overlap_synchronous_state_space,all_jsd_attractors])):
            data = np.ones((len(formalisms),len(formalisms)))
            for i in range(len(formalisms)):
                for j in range(len(formalisms)):
                    data[i,j] = corr_function(metric[i],metric[j])[0]
            
            f,ax = plt.subplots(figsize=(2.33,2.33))
            im = ax.imshow(data,origin='upper',vmin=-1,vmax=1,cmap=matplotlib.cm.RdBu)
            ax.set_yticks(range(len(formalisms)))
            ax.set_yticklabels(formalisms)
            ax.set_xticks([])
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            for kk in range(len(formalisms)):
                ax.text(kk,3.7,formalisms[kk],ha='center',va='top')
            [x1,x2] = ax.get_xlim()

            divider = make_axes_locatable(ax)
            caxax = divider.append_axes("right", size='8%', pad=0.1)    
            cbar = f.colorbar(im,cax=caxax,label=name.capitalize()+' correlation across\n100 biological networks')
        
            plt.savefig(name+'_correlation_among_formalisms_%s.pdf' % name_metric.replace(' ','_'),bbox_inches='tight')
    
                






















    #Stratify all regulatory rules in the Boolean models by their number of activators and inhibitors, as well as by 
    #the presence/absence of auto-regulation. Exclude rules that contain non-essential or conditional inputs.
    DIFFERENTIATE_AUTOREGULATION = True
    res,res_unsorted,res_f,len_res = stratify_all_regulatory_rules(Fs,Is,DIFFERENTIATE_AUTOREGULATION = DIFFERENTIATE_AUTOREGULATION)

    
    unique_res = np.array(list(set(res)))
    len_unique_res = np.array(list(map(len,unique_res)))
    dict_unique_res = dict(zip(unique_res,list(range(len(unique_res)))))


    #compute the function-level similarity between threshold and bio rules
    threshold_formalisms = [generate_threshold_rule_ising,generate_threshold_rule_ising_mod,generate_threshold_rule_01,generate_threshold_rule_01_mod]
    formalisms = ['Ising','Ising'+r'$^{*}$','01',r'01$^{*}$']
    n_threshold_formalism = len(threshold_formalisms)
    
    differences_f_to_threshold,differences_f_to_threshold_stratified,mean_difference_threshold_per_identifer,n_difference_threshold_per_identifer = compute_similarity_between_threshold_and_bio_rules(res_f,res_unsorted,res,threshold_formalisms=threshold_formalisms,formalisms=formalisms)
    
    
    #print to an Excel file the function-level similarity between threshold and bio rules
    degrees = np.arange(2,4,dtype=int)  
    dummy = []
    for degree in degrees:
        indices = np.where(len_unique_res==degree)[0]
        for index in indices:
            # print(unique_res[index],n_difference_threshold_per_identifer[0][index])
            # print(1-mean_difference_threshold_per_identifer[2][index],1-mean_difference_threshold_per_identifer[0][index],1-mean_difference_threshold_per_identifer[1][index])
            # print()
            dummy.append([unique_res[index],n_difference_threshold_per_identifer[0][index]]+[1-mean_difference_threshold_per_identifer[iii][index] for iii in range(len(threshold_formalisms))])
    A = pd.DataFrame(dummy,columns=['type','number in bio models']+formalisms)
    A.to_excel('similarity_threshold_vs_bio_rules_degree%s.xlsx' % ('_'.join(map(str,degrees))))
    
    
    #print the average proportion of 1s in biological rules of a certain degree
    bias_per_f =np.array( list(map(np.mean,res_f)))
    for degree in degrees:
        indices = np.where(len_res==degree)[0]
        print(degree,np.mean(bias_per_f[indices]))
    
    
    #print the distribution of function-level similarity between threshold and bio rules, stratified by degree
    degrees = np.arange(2,7,dtype=int)  
    f,ax = plt.subplots(figsize=(5.5,3))
    colors = ['blue','orange','green','black'] 
    for iii,generate_threshold_rule in enumerate(threshold_formalisms):
        data_full = []
        for degree in degrees:
            indices = np.where(len_unique_res==degree)[0]
            data_full.append(1-np.array(differences_f_to_threshold[iii])[len_res == degree])
        box = ax.boxplot(data_full,notch=True,positions=n_threshold_formalism*degrees-n_threshold_formalism/2-1.35+0.8*iii,showmeans=True,label=formalisms[iii])
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color=colors[iii])
        for el in (box['means']):
            el.set_markerfacecolor(colors[iii])
            el.set_markeredgecolor(colors[iii])
        for el in (box['fliers']):
            el.set_markeredgecolor(colors[iii])
        #ax.set_ylabel(data)
    ax.set_ylabel('agreement with biological rules')
    ax.set_xlabel('degree')
    xticks = n_threshold_formalism*degrees-n_threshold_formalism/2-1+0.8
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(map(str,degrees)))
    [y1,y2] = ax.get_ylim()
    for i,(d,x) in enumerate(zip(degrees,xticks)):
        count = sum(len_res==d)
        ax.text(x,y2+0.01*(y2-y1),'n='+str(count),ha='center',va='center')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='center',bbox_to_anchor=[0.5,1.12],ncol=4,frameon=False)
    plt.savefig('agreement_with_bio_rules_box.pdf',bbox_inches='tight')


    n_threshold_formalisms = len(threshold_formalisms)
    degrees = np.arange(2,9,dtype=int)  
    agreements = np.zeros((n_threshold_formalisms,len(degrees)))
    agreement_stds = np.zeros((n_threshold_formalisms,len(degrees)))
    percentile25 = np.zeros((n_threshold_formalisms,len(degrees)))
    percentile75 = np.zeros((n_threshold_formalisms,len(degrees)))
    counts = []
    for ii,degree in enumerate(degrees):
        indices = np.where(len_res==degree)[0]
        counts.append(len(indices))
        for iii,generate_threshold_rule in enumerate(threshold_formalisms):
            agreement = np.mean(1 - np.array(differences_f_to_threshold[iii])[indices])
            agreement_std = np.std(1 - np.array(differences_f_to_threshold[iii])[indices])
            percentile25[iii,ii] = np.percentile(1 - np.array(differences_f_to_threshold[iii])[indices], 25)
            percentile75[iii,ii] = np.percentile(1 - np.array(differences_f_to_threshold[iii])[indices], 75)
            print(degree,iii,np.round(agreement,3))
            agreements[iii,ii] = agreement
            agreement_stds[iii,ii] = agreement_std
    
    f,ax = plt.subplots(figsize=(5,3))
    colors = ['blue','orange','green','black'] 
    ls = ['-','--',':','-.']
    markers = ['o','x','D','D']
    for iii,generate_threshold_rule in enumerate(threshold_formalisms):
        ax.plot(degrees,agreements[iii],color=colors[iii],ls=ls[iii],marker=markers[iii],label=formalisms[iii])
    [y1,y2] = ax.get_ylim()
    for i,(d,c) in enumerate(zip(degrees,counts)):
        ax.text(d,y2+0.03*(y2-y1),'n='+str(c),ha='center',va='center')
    ax.set_xlabel('degree')
    ax.set_xlim(ax.get_xlim()+np.array([-0.5,0.5]))
    ax.spines[['top','right']].set_visible(False)
    yticks = np.linspace(0,1,21)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(el*100))+'%' for el in yticks])
    ax.set_ylim([y1,y2+0.06*(y2-y1)])
    ax.set_ylabel('mean agreement with biological rules')
    ax.legend(loc='best',frameon=False)
    plt.savefig('agreement_with_bio_rules.pdf',bbox_inches='tight')

        





    #compute and plot the expected against the observed number of regulatory functions with a specific number of
    #activators and inhibitors, do not differentiate by auto-regulation
    DIFFERENTIATE_AUTOREGULATION = False
    res,res_unsorted,res_f,len_res = stratify_all_regulatory_rules(Fs,Is,DIFFERENTIATE_AUTOREGULATION = DIFFERENTIATE_AUTOREGULATION)
    differences_f_to_threshold,differences_f_to_threshold_stratified,mean_difference_threshold_per_identifer,n_difference_threshold_per_identifer = compute_similarity_between_threshold_and_bio_rules(res_f,res_unsorted,res,threshold_formalisms=threshold_formalisms,formalisms=formalisms)

    unique_res = np.array(list(set(res)))
    len_unique_res = np.array(list(map(len,unique_res)))
    
    degrees = np.arange(2,7,dtype=int)  
    f,ax = plt.subplots(len(degrees),1,sharex=True,figsize=(5,6))
    values = [0.5,1,2,4]
    REVERSE=True
    TEXT_BOTTOM = True

    import matplotlib
    cmap = matplotlib.cm.jet
    colors = [cmap(el*1./max(degrees)) for el in range(max(degrees)+1)]

    for ii,degree in enumerate(degrees):
        indices = np.where(len_unique_res==degree)[0]  
        n_activators = np.array([el.count('a') for el in unique_res[indices]])
        sorted_indices = np.argsort(n_activators)
        indices = indices[sorted_indices]
        n_activators = n_activators[sorted_indices]
        observed_count = n_difference_threshold_per_identifer[0][indices]
        n_total_per_degree = sum(observed_count)
        observed_proportions = observed_count/n_total_per_degree
        proportion_activators = np.dot(n_activators,observed_proportions)/degree
        expected_proportions = np.array([proportion_activators**el * (1-proportion_activators)**(degree-el) * scipy.special.binom(degree,el) for el in n_activators])
        print(degree,proportion_activators,max(observed_proportions/expected_proportions - 1),min(observed_proportions/expected_proportions - 1))
        
        if REVERSE:
            expected_proportions=expected_proportions[::-1]
            observed_proportions=observed_proportions[::-1]
            observed_count=observed_count[::-1]
        
        # Calculate the height of each bar as the difference from the expected value
        bar_heights = observed_proportions/expected_proportions - 1
        
        # Plotting
        bars = ax[ii].bar(n_activators, bar_heights, bottom=1, color=colors[:degree+1])
        ax[ii].spines[['top','right']].set_visible(False)
        ax[ii].semilogy([-100,100],[1,1],'k:',lw=1)
        #plt.yscale('log')  # Set y-axis to log scale
        y1,y2 = ax[ii].get_ylim()
        if TEXT_BOTTOM:
            ax[ii].set_ylim([np.exp(np.log(y1)-0.3*(np.log(y2)-np.log(y1))),y2])
            for jj,el in enumerate(observed_count):
                ax[ii].text(jj,np.exp(np.log(y1)-0.25*(np.log(y2)-np.log(y1))),'n = '+str(el),ha='center',va='bottom',fontsize=8)
            ax[ii].text(-0.7,y2,'degree = '+str(degree),ha='left',va='bottom',fontsize=8)
        else:
            for jj,el in enumerate(observed_count):
                ax[ii].text(jj,y2,'n = '+str(el),ha='center',va='bottom',fontsize=8)
        y1,y2 = ax[ii].get_ylim()
        yticks = []
        for val in values:
            if val>y1 and val<y2:
                yticks.append(val)
        if len(yticks)==1:
            yticks.append( np.floor(y2*10)/10 )
        ax[ii].yaxis.set_minor_locator(MultipleLocator(9))
        ax[ii].set_yticks(yticks)
        ax[ii].set_yticklabels(list(map(str,yticks)))


        if degree<degrees[-1]:
            ax[ii].tick_params(axis='x',length=0)
        ax[ii].set_xlim([-0.7,degrees[-1]+0.7])
    
    ax[-1].set_xlabel('number of '+('negative' if REVERSE else 'positive')+' regulators')
    ax[-1].set_ylabel('ratio observed/expected functions')
    ax[-1].yaxis.set_label_coords(0.05, 0.5, transform=f.transFigure)
    min_ylim = min([ax[ii].get_ylim()[0] for ii,degree in enumerate(degrees)])
    max_ylim = max([ax[ii].get_ylim()[1] for ii,degree in enumerate(degrees)])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig('ratio_observed_expected_functions.pdf',bbox_inches='tight')
        





    PROPORTION=True
    REVERSE=True
    height_rectangle = 0.1
    f,ax = plt.subplots(figsize=(5,3))
    colors = [cmap(el*1./max(degrees)) for el in range(max(degrees)+1)]
    width = 0.62
    epsilon = 0.25
    lw=1
    for k,degree in enumerate(degrees):
        indices = np.where(len_unique_res==degree)[0]  
        n_activators = np.array([el.count('a') for el in unique_res[indices]])
        sorted_indices = np.argsort(n_activators)
        indices = indices[sorted_indices]
        n_activators = n_activators[sorted_indices]
        observed_count = n_difference_threshold_per_identifer[0][indices]
        n_total_per_degree = sum(observed_count)
        observed_proportions = observed_count/n_total_per_degree
        proportion_activators = np.dot(n_activators,observed_proportions)/degree
        expected_proportions = np.array([proportion_activators**el * (1-proportion_activators)**(degree-el) * scipy.special.binom(degree,el) for el in n_activators])
        
        if REVERSE:
            expected_proportions=expected_proportions[::-1]
            observed_proportions=observed_proportions[::-1]
        
        ax.text(2*k,1.05,'n=%i' % n_total_per_degree,va='center',ha='center',clip_on=False )

        s=0
        for el in range(degree+1):
            ax.bar([2*k-width/4-epsilon],[expected_proportions[el]],color=colors[el],bottom=[s],width=width)
            s+=expected_proportions[el]
        s=0
        for el in range(degree+1):
            ax.bar([2*k+width/4+epsilon],[observed_proportions[el]],color=colors[el],bottom=[s],width=width)
            s+=observed_proportions[el]
    for k,degree in enumerate(degrees):
        for y in [0,1]:
            ax.plot([2*k-3*width/4-epsilon,2*k-3*width/4-epsilon+width],[y,y],'k--',lw=lw)
            ax.plot([2*k-width/4+epsilon,2*k-width/4+epsilon+width],[y,y],'k-',lw=lw)
        for dx in [0,width]:
            ax.plot([2*k-3*width/4-epsilon+dx,2*k-3*width/4-epsilon+dx],[0,1],'k--',lw=lw)
            ax.plot([2*k-width/4+epsilon+dx,2*k-width/4+epsilon+dx],[0,1],'k-',lw=lw)   
    x1,x2 = ax.get_xlim()
    ax.text(1.4,-0.25,'number of negative regulators' if REVERSE else 'number of positive regulators',ha='center',va='center')
    for j in range(degree+1):
        ax.add_patch(matplotlib.patches.Rectangle([-2.4+2*(j//2),-0.45 if j%2==0 else -0.61],1,height_rectangle,color=colors[j],clip_on=False))
        ax.text(-1.1+2*(j//2),(-0.45 if j%2==0 else -0.61)+height_rectangle/2.5,str(j),ha='left',va='center',clip_on=False)
    
    for j,(ls,label) in enumerate(zip(['--','-'],['expected','observed'])):
        ax.text(7,-0.45-0.16*j+height_rectangle/2.5,label,ha='left',va='center',clip_on=False)
        ax.plot([5.7,6.7],[-0.45-0.16*j,-0.45-0.16*j],'k',ls=ls,lw=1,clip_on=False)
        ax.plot([5.7,6.7],np.array([-0.45-0.16*j,-0.45-0.16*j])+height_rectangle,'k',ls=ls,lw=1,clip_on=False)
        ax.plot([5.7,5.7],np.array([-0.45-0.16*j,-0.45-0.16*j+height_rectangle]),'k',ls=ls,lw=1,clip_on=False)
        ax.plot([6.7,6.7],np.array([-0.45-0.16*j,-0.45-0.16*j+height_rectangle]),'k',ls=ls,lw=1,clip_on=False)
    ax.set_ylim([0,1])
    ax.set_xlim([x1,x2])
    
    ax.set_xticks(np.arange(0,2*k+1,2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels([str(degree) for k,degree in enumerate(degrees)])
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel('degree')
    ax.set_ylabel('proportion of functions')
    plt.savefig('ratio_observed_expected_functions_nice.pdf',bbox_inches='tight')

            
#Compute the similarities among different threshold formalisms
types = np.array(['decreasing','increasing'])
mean_similarities_ising_01 = []
mean_similarities_ising_01_mod = []
mean_similarities_01_01_mod = []
mean_similarities_ising_ising_mod = []
mean_similarities_01_ising_mod = []
mean_similarities_01_mod_ising_mod = []

PRINT=True
for degree in range(2,3):
    
    function_types = []
    f_isings = []
    f_ising_mods  = []
    f_01s = []
    f_01_mods = []
    for which in list(itertools.combinations_with_replacement([0,1],degree)):
        types_regulations = types[np.array(which)][::-1]
        f_ising = generate_threshold_rule_ising([],type_of_each_regulation=types_regulations)[0]
        f_ising_mod = generate_threshold_rule_ising_mod([],type_of_each_regulation=types_regulations)[0]
        f_01 = generate_threshold_rule_01([],type_of_each_regulation=types_regulations)[0]
        f_01_mod = generate_threshold_rule_01_mod([],type_of_each_regulation=types_regulations)[0]
        if PRINT:
            print(False,types_regulations)
            print(f_ising)
            print(f_ising_mod)
            print(f_01)
            print(f_01_mod)
            print()
        function_types.append([False,types_regulations])
        f_isings.append(f_ising)
        f_ising_mods.append(f_ising_mod)
        f_01s.append(f_01)
        f_01_mods.append(f_01_mod)
    
        print(can.get_edge_effectiveness(f_ising))
        print(can.get_edge_effectiveness(f_ising_mod))
        print(can.get_edge_effectiveness(f_01))
        print(can.get_edge_effectiveness(f_01_mod))
        print()
        print()
        
    #autoregulation at index 0
    for which in list(itertools.combinations_with_replacement([0,1],degree-1)):
        for auto_reg_type in types:
            types_regulations = np.append(auto_reg_type,types[np.array(which)][::-1])
            f_ising = generate_threshold_rule_ising([],AUTOREGULATION_INDEX=0,type_of_each_regulation=types_regulations)[0]
            f_ising_mod = generate_threshold_rule_ising_mod([],AUTOREGULATION_INDEX=0,type_of_each_regulation=types_regulations)[0]
            f_01 = generate_threshold_rule_01([],AUTOREGULATION_INDEX=0,type_of_each_regulation=types_regulations)[0]
            f_01_mod = generate_threshold_rule_01_mod([],AUTOREGULATION_INDEX=0,type_of_each_regulation=types_regulations)[0]
            if PRINT:
                print(True,types_regulations)
                print(f_ising)
                print(f_ising_mod)
                print(f_01)
                print(f_01_mod)
                print()
            function_types.append([True,types_regulations])
            f_isings.append(f_ising)
            f_ising_mods.append(f_ising_mod)
            f_01s.append(f_01)
            f_01_mods.append(f_01_mod)
        
            print(can.get_edge_effectiveness(f_ising))
            print(can.get_edge_effectiveness(f_ising_mod))
            print(can.get_edge_effectiveness(f_01))
            print(can.get_edge_effectiveness(f_01_mod))
            print()
            print()
    
    print('\nIsing vs 01')
    similarities_ising_01 = []
    for t,f1,f2 in zip(function_types,f_isings,f_01s):
        similarity = get_similarity_functions(f1,f2)
        similarities_ising_01.append(similarity)
        if PRINT:
            print(t,similarity)
        
    print('\nIsing vs 01_mod')
    similarities_ising_01_mod = []
    for t,f1,f2 in zip(function_types,f_isings,f_01_mods):
        similarity = get_similarity_functions(f1,f2)
        similarities_ising_01_mod.append(similarity)
        if PRINT:
            print(t,similarity)    
            
    print('\n01 vs 01_mod')
    similarities_01_01_mod = []
    for t,f1,f2 in zip(function_types,f_01s,f_01_mods):
        similarity = get_similarity_functions(f1,f2)
        similarities_01_01_mod.append(similarity)
        if PRINT:
            print(t,similarity)  
    
    print('\nIsing vs Ising_mod')
    similarities_ising_ising_mod = []
    for t,f1,f2 in zip(function_types,f_isings,f_ising_mods):
        similarity = get_similarity_functions(f1,f2)
        similarities_ising_ising_mod.append(similarity)
        if PRINT:
            print(t,similarity)

    print('\n01 vs Ising_mod')
    similarities_01_ising_mod = []
    for t,f1,f2 in zip(function_types,f_01s,f_ising_mods):
        similarity = get_similarity_functions(f1,f2)
        similarities_01_ising_mod.append(similarity)
        if PRINT:
            print(t,similarity)

    print('\n01_mod vs Ising_mod')
    similarities_01_mod_ising_mod = []
    for t,f1,f2 in zip(function_types,f_01_mods,f_ising_mods):
        similarity = get_similarity_functions(f1,f2)
        similarities_01_mod_ising_mod.append(similarity)
        if PRINT:
            print(t,similarity)            
            
            
            
    mean_similarities_ising_01.append(np.mean(similarities_ising_01))
    mean_similarities_ising_01_mod.append(np.mean(similarities_ising_01_mod))
    mean_similarities_01_01_mod.append(np.mean(similarities_01_01_mod))
    mean_similarities_ising_ising_mod.append()
            
    
    
