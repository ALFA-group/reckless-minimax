"""
Implements the algorithm introduced in -
@Article{qiu2017new,
  Title                    = {A New Differential Evolution Algorithm for Minimax Optimization in Robust Design},
  Author                   = {Qiu, Xin and Xu, Jian-Xin and Xu, Yinghao and Tan, Kay Chen},
  Journal                  = {IEEE Transactions on Cybernetics},
  Year                     = {2017},
  Publisher                = {IEEE}
Variables names used in this code conform to the notation used in the above paper
"""

from __future__ import print_function, division
import numpy as np
import random
# Import classes from coev.py
from coev import Individual, Population, Coev
# Import constants from coev.py
from coev import MINIMIZER, MAXIMIZER, POPULATION_NAMES, SORT_ORDERS, DEFAULT_FITNESSES
import heapq
import copy

# for debugging
def print_population(population_obj):
    for k,v in population_obj.iteritems():
        print(k+"::", end="\n")
        print(v, end="\n\n")

# for debugging
def print_heap(lst):
    print('\n'.join("{}\n{}".format(k[0], k[1]) for k in lst), end='\n\n')

class MMDE(Coev):
    def __init__(self, fct, D_x, D_y, max_fevals, seed=1, K_s = 190, F=0.7, Cr=0.5, T_partial_regeneration=10, mutation_probability=0.9
                 , population_size=100, tournament_size=2, n_replacements=1, verbose=False, regeneration_probability=0.0):

        self.K_s = K_s
        self.F = F
        self.Cr = Cr
        self.regeneration_probability = regeneration_probability
        self.T_partial_regeneration = T_partial_regeneration
        # max_runs: back calculates how many times run() must execute to be within max_feval
        self.max_runs = max(1, max_fevals // (K_s + population_size + int(round((regeneration_probability * T_partial_regeneration)))))
        super(MMDE, self).__init__(fct, D_x, D_y, mutation_probability, population_size, tournament_size, n_replacements, max_fevals, seed, verbose)
        # a workaround for working with lists
        self._fct = lambda x,y: fct(np.asarray(x), np.asarray(y))

    def run(self):
        population = initialize_population(self.population_size, self.individual_sizes)
        population = prepare_population_list(population)
        
        for _ in range(self.max_runs):
            population_heapified_list = heapify_population(population, self._fct)
            sorted_population_list = bottom_boosting_scheme(population_heapified_list, self.K_s, self.F, self.Cr, self._fct)
            population = regeneration_strategy(sorted_population_list, self.T_partial_regeneration, self.F, self.Cr, self.regeneration_probability, self._fct)
       
        # calculate best population from best_populations_sorted_list
        bestX = population[0][1][0]
        bestS = population[0][1][1]
        bestFnEval = population[0][0]
        return np.asarray(bestX), np.asarray(bestS), np.asarray(bestFnEval)


def initialize_population(population_size, individual_sizes):
    population = {}
    for population_name in POPULATION_NAMES:
        population[population_name] = Population(individuals=[],
                                                  name=population_name,
                                                  sort_order=SORT_ORDERS[population_name],
                                                  default_fitness=DEFAULT_FITNESSES[population_name]
                                                  )
        for _ in range(population_size):
            solution = Individual(genome=[random.random() for _ in range(individual_sizes[population_name])],
                                  fitness=population[population_name].default_fitness
                                  )
            population[population_name].individuals.append(solution)

    #print_population(population)
    return population

def prepare_population_list(population):
    population_list = []
    for i in range(len(population[MINIMIZER].individuals)):
        population_list.append((None,
                                (population[MINIMIZER].individuals[i].genome, 
                                 population[MAXIMIZER].individuals[i].genome
                                )
                                ))
    return population_list

def heapify_population(population, fct):
    list_evaluations = []
    for i in range(len(population)):
        evaluation = fct(population[i][1][0],
                          population[i][1][1]
                         )
        # The data structure being stored in the heap is
        # a pair of a pair.
        # Specifically, (fct(X,S), ([X], [S])) 
        # where X.shape: 1xdims of min variable; S.shape: 1xdims of max variable
        # Specifically, (fct(X,S), ([X], [S]))
        # where X: dims of min variable; S: dims of max variable
        list_evaluations.append((evaluation,
                                (population[i][1][0],
                                 population[i][1][1]
                                )
                                ))

    heapq.heapify(list_evaluations)
    #test heap
    #print_heap([heapq.heappop(list_evaluations) for _ in range(len(list_evaluations))])
    return list_evaluations

def bottom_boosting_scheme(population_list, K_s, F, Cr, fct):
    fn_evals = 0
    while fn_evals < K_s:
        min_eval = population_list[0][0]
        min_X = population_list[0][1][0]
        min_S = population_list[0][1][1]
        random_ind = random.sample(range(1, len(population_list)), 3)
        S_new = generate_mutant_bottom_boosting(population_list, random_ind, F)
        trial = binomial_recombination(min_S, S_new, Cr)

        ev1 = fct(min_X, trial)
        if ev1 > min_eval:
            len_orig = len(population_list)
            heapq.heappop(population_list)
            assert len(population_list) == len_orig -1 
            list_to_add = (ev1,(min_X, trial))
            heapq.heappush(population_list, list_to_add)
            #print_heap(population_list)  

        fn_evals = fn_evals + 1

    sorted_population_list = [heapq.heappop(population_list) for i in range(len(population_list))]
    
    for i in range(len(sorted_population_list) - 1):
        assert sorted_population_list[i][0] <= sorted_population_list[i+1][0]
        
    return sorted_population_list

def generate_mutant_bottom_boosting(population_list, random_ind, F):
    M1 = population_list[random_ind[0]][1][1]
    M2 = population_list[random_ind[1]][1][1]
    M3 = population_list[random_ind[2]][1][1]
    M_new = (np.asarray(M1) + F*(np.asarray(M2)-np.asarray(M3)))
    # Need to scale to [0,1], since that is what Coev expects
    # Actual range of problem domain is handled by the Sandle_point classes
    # Using min-max scaling
    # min: M_new = 0 + F(0-1) = -F
    # max: M_new = 1 + F*(1-0) = 1 + F
    sz = M_new.shape[0]
    M_new = _scale_values(M_new, (-1*F)*np.ones(sz),(1+F)*np.ones(sz), np.zeros(sz), np.ones(sz))
    return M_new.tolist()


def binomial_recombination(M_orig, M_mutated, crossover_prob):
    if not len(M_orig) > 3:
        j_rand = -1
    else:
        j_rand = random.sample(range(0, len(M_orig)), 1)[0]
    for j in range(len(M_orig)):
        rand_prob = random.random()
        if (rand_prob <= crossover_prob) or (j == j_rand):
            M_orig[j] = M_mutated[j]
    return M_orig

def regeneration_strategy(population_list, T_partial_regeneration, F, Cr, regeneration_probability, fct):
    assert T_partial_regeneration < len(population_list)
    # meaningless randomization of indices until T!
    list_of_ind_T = random.sample(range(0, T_partial_regeneration), T_partial_regeneration)
    t = 0
    while t < T_partial_regeneration:
        i = list_of_ind_T[t]
        random_ind = random.sample(range(0, len(population_list)), 2)
        while i in random_ind:
            random_ind = random.sample(range(0, len(population_list)), 2)
        if random.random() < regeneration_probability:
            X_new = generate_mutant_regeneration(population_list, random_ind, F, population_list[i][1][0])
            trial = binomial_recombination(population_list[i][1][0], X_new, Cr)
            # wow to the amount of randomness
            ind_to_replace = len(population_list) - t - 1
            S_random = [random.random() for _ in range(len(trial))]
            new_tuple = (fct(trial, S_random), (trial, S_random))        
            population_list[ind_to_replace] = copy.deepcopy(new_tuple)
        t = t+1
        
    return population_list

def _scale_values(values, old_lb, old_ub, new_lb, new_ub):
    sz = values.shape[0]
    for idx in range(sz):
        assert (values[idx] <= old_ub[idx] and values[idx] >= old_lb[idx])
    new_values = new_lb + (new_ub - new_lb) * (values - old_lb) / (old_ub - old_lb)
    for idx in range(sz):
        assert (new_values[idx] <= new_ub[idx] and new_values[idx] >= new_lb[idx])
    return new_values
    

def generate_mutant_regeneration(population_list, random_ind, F, X_i):
    M1 = population_list[random_ind[0]][1][0]
    M2 = population_list[random_ind[1]][1][0]
  
    assert random_ind[0] != random_ind[1]
    assert M1 != M2

    M_new = (np.asarray(X_i) + F*(np.asarray(M1)-np.asarray(M2)))
    # Need to scale to [0,1], since that is what Coev expects
    # Actual range of problem domain is handled by the Sandle_point classes
    # Using min-max scaling
    # min: M_new = 0 + F(0-1) = -F
    # max: M_new = 1 + F*(1-0) = 1 + F
    #M_new = (M_new + F) /(1+(2*F))
    sz = M_new.shape[0]
    M_new = _scale_values(M_new, (-1*F)*np.ones(sz),(1+F)*np.ones(sz), np.zeros(sz), np.ones(sz))
    return M_new.tolist()

if __name__ == '__main__':
    from test_suite.toy_problem import ToyProblem
    from test_suite.robust_de_problems import RobustDEProblem

    
    _D_x = 2
    _D_y = 2
    tp = ToyProblem(D_x=_D_x, D_y=_D_y)
    _fct = tp.evaluate
    _mutation_probability = 0.9
    _population_size = 100
    _tournament_size = 2
    _n_replacements = 1
    _max_feval = 20000
    _verbose = True
    # DE operators
    _F = 0.7 # Scaling factor. Real value in [0,1]
    _cross_over_probability = 0.5
    _regeneration_probability = 0.01
    
    _K_s = 10 # Number of function evaluations
    _T_partial_regeneration = 10 

    

    _D_x = 1
    _D_y = 1
    _max_feval = 20000
    _verbose = False
    tp = RobustDEProblem(D_x=_D_x, D_y=_D_y, fun_num=1)
    _fct = tp.evaluate


    mmde_obj2 = MMDE(_fct,
                     _D_x, _D_y,max_fevals=_max_feval, verbose=_verbose, seed=3)

    x_opt, y_opt, f_opt = mmde_obj2.run()
    print("Found: x_opt: {}, y_opt: {}, fevals:{}".format(x_opt, y_opt, tp.get_num_fevals()))
    print("Saddle-pt: minimizer: {}, maxmimizer: {}".format(tp.get_x_opt(), tp.get_y_opt()) )
    print("Found: minimizer: {}, maxmimizer: {}".format(x_opt, y_opt))
    print("Obj values at sp {}, found {}".format(tp.evaluate(tp.get_x_opt(), tp.get_y_opt()), tp.evaluate(x_opt, y_opt)), f_opt)
    print("relative robustness measure: {}".format(tp.relative_robustness(x_opt, y_opt)))
    print("mse measure: {}".format(tp.mse(x_opt, y_opt)))