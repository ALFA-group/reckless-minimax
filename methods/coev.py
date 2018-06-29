import random
import numpy as np
from methods.min_max_method import MinMaxMethod

DEFAULT_FITNESS_MINIMIZER = float('-inf')
DEFAULT_FITNESS_MAXIMIZER = float('inf')
MINIMIZER = 'minimizer'
MAXIMIZER = 'maximizer'
DEFAULT_FITNESSES = {MAXIMIZER: DEFAULT_FITNESS_MAXIMIZER,
                     MINIMIZER: DEFAULT_FITNESS_MINIMIZER,
                     }
POPULATION_NAMES = (MINIMIZER, MAXIMIZER)
SORT_ORDERS = {MAXIMIZER: True,
               MINIMIZER: False}


class Individual(object):

    def __init__(self, genome, fitness):
        self.genome = genome
        self.fitness = fitness
        self.adversary_solution = None

    def __str__(self):
        return "y={} x={} xA={}".format(self.fitness, self.genome, self.adversary_solution)


class Population(object):

    def __init__(self, sort_order, individuals, name, default_fitness):
        self.sort_order = sort_order
        self.individuals = individuals
        self.name = name
        self.default_fitness = default_fitness
        if self.sort_order:
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def sort_population(self):
        self.individuals.sort(key=lambda x: x.fitness, reverse=self.sort_order)

    def replacement(self, new_population, n_replacements=1):
        new_population.sort_population()
        self.sort_population()
        # TODO break out early
        for i in range(n_replacements):
            j = i - n_replacements
            if self.compare(self.individuals[j].fitness, new_population.individuals[i].fitness):
                self.individuals[j] = new_population.individuals[i]

    def __str__(self):
        return "{} {}".format(self.name, ', '.join(map(str, self.individuals)))


class Coev(MinMaxMethod):


    def __init__(self, fct, D_x, D_y , mutation_probability=0.9, population_size=10, tournament_size=2,
                 n_replacements=1, max_fevals = 100, seed=1, verbose=False):
        super(Coev, self).__init__(fct, D_x, D_y, max_fevals=max_fevals, seed=seed)
        self.verbose = verbose
        self.T = max_fevals // (population_size ** 2)
        if self.verbose:
            print("{} T: {}".format(self, self.T))

        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.n_replacements = n_replacements
        self.individual_sizes = {MINIMIZER: self._D_x, MAXIMIZER: self._D_y}

    def run(self):
        raise NotImplementedError


class CoevAlternating(Coev):

    def __init__(self, fct, D_x, D_y , mutation_probability=0.9, population_size=10, tournament_size=2, n_replacements=1, max_fevals = 100, seed=1, verbose = False):
        super(CoevAlternating, self).__init__(fct, D_x, D_y , mutation_probability, population_size, tournament_size, n_replacements, max_fevals, seed,verbose)
        self.reverse_population_names = list(POPULATION_NAMES[:])
        self.reverse_population_names.reverse()

    def run(self):
        populations = initialize_populations(self.population_size, self.individual_sizes)

        t = 0
        evaluate_fitness(populations, self._fct)
        if self.verbose:
            for population_name in POPULATION_NAMES:
                print("t:{} {} best:{}".format(t, population_name, populations[population_name].individuals[0]))

        t += 1

        while t < self.T:
            # Alternate between minimizer and maximizer populations
            for attacker, defender in (POPULATION_NAMES, self.reverse_population_names):
                if t >= self.T:
                    break

                new_population = tournament_selection(
                    populations[attacker],
                    self.population_size,
                    self.tournament_size,
                )
                mutate_gaussian(new_population, self.mutation_probability)
                alternating_populations = {attacker: new_population,
                                           defender: populations[defender]}
                evaluate_fitness(alternating_populations, self._fct)

                # Replace the worst with the best new
                populations[attacker].replacement(new_population, self.n_replacements)
                # Print best
                populations[attacker].sort_population()
                if self.verbose:
                    print("t:{} {} best:{}".format(t, attacker, populations[attacker].individuals[0]))


                t += 1

        return np.array(populations[MINIMIZER].individuals[0].genome), np.array(
            populations[MINIMIZER].individuals[0].adversary_solution), populations[MINIMIZER].individuals[0].fitness


class CoevParallel(Coev):

    def run(self):
        populations = initialize_populations(self.population_size, self.individual_sizes)

        t = 0
        evaluate_fitness(populations, self._fct)
        if self.verbose:
            for population_name in POPULATION_NAMES:
                print("t:{} {} best:{}".format(t, population_name, populations[population_name].individuals[0]))

        t += 1

        while t < self.T:
            new_populations = {}
            for population_name in populations.keys():
                new_populations[population_name] = tournament_selection(
                    populations[population_name],
                    self.population_size,
                    self.tournament_size,
                )
                mutate_gaussian(new_populations[population_name], self.mutation_probability)

            evaluate_fitness(new_populations, self._fct)

            for population_name in populations.keys():
                populations[population_name].replacement(new_populations[population_name], self.n_replacements)
                # Print best
                populations[population_name].sort_population()
                if self.verbose:
                    print("t:{} {} best:{}".format(t, population_name, populations[population_name].individuals[0]))

            t += 1


        return np.array(populations[MINIMIZER].individuals[0].genome), np.array(
            populations[MINIMIZER].individuals[0].adversary_solution), populations[MINIMIZER].individuals[0].fitness


def initialize_populations(population_size, individual_sizes):
    populations = {}
    for population_name in POPULATION_NAMES:
        populations[population_name] = Population(individuals=[],
                                                  name=population_name,
                                                  sort_order=SORT_ORDERS[population_name],
                                                  default_fitness=DEFAULT_FITNESSES[population_name]
                                                  )
        for _ in range(population_size):
            solution = Individual(genome=[random.random() for _ in range(individual_sizes[population_name])],
                                  fitness=populations[population_name].default_fitness
                                  )
            populations[population_name].individuals.append(solution)

    return populations


def mutate_gaussian(new_population, mutation_probability):
    for i in range(len(new_population.individuals)):
        for j in range(len(new_population.individuals[0].genome)):
            if random.random() < mutation_probability:
                # Clip genome to [0, 1]
                #new_population.individuals[i].genome[j] = + max(0.0, min(1.0, random.gauss(0, 1)))
                new_population.individuals[i].genome[j] = np.clip(new_population.individuals[i].genome[j] + random.gauss(0, 1), 0, 1)

def tournament_selection(population, population_size, tournament_size):
    assert 0 < tournament_size <= len(population.individuals), "Invalid tournament size: {}".format(tournament_size)

    competition_population = Population(sort_order=population.sort_order,
                                        individuals=[],
                                        default_fitness=population.default_fitness,
                                        name="competition")
    new_population = Population(sort_order=population.sort_order,
                                individuals=[],
                                default_fitness=population.default_fitness,
                                name=population.name
                                )

    # Iterate until there are enough tournament winners selected
    while len(new_population.individuals) < population_size:
        # Randomly select tournament size individual solutions
        # from the population.
        competitors = random.sample(population.individuals, tournament_size)
        competition_population.individuals = competitors
        # Rank the selected solutions
        competition_population.sort_population()
        # Copy the solution
        winner = Individual(genome=competitors[0].genome[:],
                            fitness=competition_population.default_fitness)
        # Append the best solution to the winners
        new_population.individuals.append(winner)
    assert len(new_population.individuals) == population_size

    return new_population


def evaluate_fitness(populations, fct):
    for i in range(len(populations[MINIMIZER].individuals)):
        for j in range(len(populations[MAXIMIZER].individuals)):

            fitness = fct(np.array(populations[MINIMIZER].individuals[i].genome),
                          np.array(populations[MAXIMIZER].individuals[j].genome))
            # Best worst case solution
            if fitness > populations[MINIMIZER].individuals[i].fitness:
                populations[MINIMIZER].individuals[i].fitness = fitness
                populations[MINIMIZER].individuals[i].adversary_solution = populations[MAXIMIZER].individuals[j].genome

            if fitness < populations[MAXIMIZER].individuals[j].fitness:
                populations[MAXIMIZER].individuals[j].fitness = fitness
                populations[MAXIMIZER].individuals[j].adversary_solution = populations[MINIMIZER].individuals[i].genome


if __name__ == '__main__':
    from test_suite.toy_problem import ToyProblem


    _mutation_probability = 0.9
    _population_size = 10
    _tournament_size = 2
    _n_replacements = 1
    _verbose = True

    max_feval = 200


    coev_algs = {'coev_parallel': CoevParallel,
                 'coev_alternating': CoevAlternating,
                 }
    for coev_alg_name, coev_alg in coev_algs.items():
        print('{}'.format(coev_alg_name))
        D_x = 2
        D_y = 2
        tp = ToyProblem(D_x=D_x, D_y=D_y)
        _fct = tp.evaluate

        alg = coev_alg(_fct, D_x, D_y, _mutation_probability, _population_size, _tournament_size, _n_replacements,
                       max_feval, _verbose)
        x_opt, y_opt, f_opt = alg.run()
        assert tp.get_num_fevals() == max_feval, "{} != {}".format(max_feval, tp.get_num_fevals())

        print("Saddle-pt: minimizer: {}, maxmimizer: {}".format(tp.get_x_opt(), tp.get_y_opt()))
        print("Found: x_opt: {}, y_opt: {}, fevals:{}".format(x_opt, y_opt, tp.get_num_fevals()))
        print("Obj values at sp {}, found {}".format(tp.evaluate(tp.get_x_opt(), tp.get_y_opt()),
                                                     tp.evaluate(x_opt, y_opt)), f_opt)
        print("relative robustness measure: {}".format(tp.relative_robustness(x_opt, y_opt)))
        print("relative loss measure: {}".format(tp.relative_loss(x_opt, y_opt)))
        print("mse measure: {}".format(tp.mse(x_opt, y_opt)))
