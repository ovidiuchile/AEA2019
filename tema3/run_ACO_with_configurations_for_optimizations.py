import os
import sys
import json
import time
import random
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from deap import creator
from deap import base
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ACO_read_data():
    global N_CITIES, CITIES_MATRIX
    N_CITIES = None
    CITIES_MATRIX = None
    if len(sys.argv) == 1:
        print('Specifiy an input file.\ne.g.: %s inputs/example.json' % os.path.basename(sys.argv[0]))
        exit(1)

    input_fp = sys.argv[1]
    with open(input_fp, 'rt') as f:
        input_data = json.load(f)
    N_CITIES = input_data["n"]
    CITIES_MATRIX = np.array(input_data['matrix'])


def ACO_dist(a, b):
    _dist = CITIES_MATRIX[a, b]
    if _dist < 1e-10:
        return 1e-10
    return _dist


def ACO_distance(last_city_visited_by_truck, city, last_drone_visit, drone_visit, drone_distance, truck_distance,
                 last_city_visited_by_drone, cities_involved):
    # cities_involved = 1
    if not last_drone_visit and not drone_visit:
        _dist = (max(drone_distance, truck_distance) + ACO_dist(last_city_visited_by_truck, city))
    elif not last_drone_visit and drone_visit:
        _dist = max(drone_distance + ACO_dist(last_city_visited_by_drone, city), truck_distance)
    elif last_drone_visit and not drone_visit:
        _dist = max(drone_distance, truck_distance + ACO_dist(last_city_visited_by_truck, city))
    else:  # if last_drone_visit and drone_visit:
        _dist = max(drone_distance + ACO_dist(last_city_visited_by_drone, city),
                    truck_distance + ACO_dist(last_city_visited_by_truck, city))
    return _dist / cities_involved


def ACO_cost(permutation, drone_visits):
    is_drone_at_a_customer = False
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    truck_distance = 0
    drone_distance = 0

    # print('cities:', [city + 1 for city in permutation])
    # i = 0

    total_time_taken = 0
    for city, visited_by_drone in zip(permutation, drone_visits):
        # print('i:', i, ', city:', city)
        # i += 1

        if not visited_by_drone:
            if is_drone_at_a_customer:
                truck_distance += ACO_dist(last_city_visited_by_truck, city)
                last_city_visited_by_truck = city
            else:
                total_time_taken += ACO_dist(last_city_visited_by_truck, city)
                # print('totaltime:', total_time_taken)
                # print()
                last_city_visited_by_truck = city
        else:
            if not is_drone_at_a_customer:
                is_drone_at_a_customer = True
                drone_distance += ACO_dist(last_city_visited_by_truck, city)
                last_city_visited_by_drone = city
            else:
                is_drone_at_a_customer = False
                drone_distance += ACO_dist(last_city_visited_by_drone, city)
                truck_distance += ACO_dist(last_city_visited_by_truck, city)
                last_city_visited_by_truck = city
                last_city_visited_by_drone = city
                total_time_taken += max(drone_distance, truck_distance)
                # print('totaltime:', total_time_taken)
                # print()
                drone_distance = 0
                truck_distance = 0

    if is_drone_at_a_customer:
        drone_distance += ACO_dist(last_city_visited_by_drone, 0)
        truck_distance += ACO_dist(last_city_visited_by_truck, 0)
        total_time_taken += max(drone_distance, truck_distance)
        # print('totaltime:', total_time_taken)
    else:
        total_time_taken += ACO_dist(last_city_visited_by_truck, 0)

    # print('finaltotaltime:', total_time_taken)
    # exit(1)

    return total_time_taken


def ACO_random_permutation(n):
    cities = list(range(1, n))
    random.shuffle(cities)
    cities = [0] + cities
    return cities


def ACO_random_bool_list(n):
    return [0 if random.randint(1, 100) <= 50 else 1 for i in range(1, n)]


def ACO_initialise_pheromone_matrix(init_pher):
    return [[init_pher for _ in range(N_CITIES)] for _ in range(N_CITIES)]


def ACO_calculate_choices(last_city_visited_by_truck, last_drone_visit, last_city_visited_by_drone, drone_distance,
                          truck_distance, cities_involved, exclude, pheromone, c_heur, c_hist):
    choices = []
    for city in range(1, N_CITIES):
        for drone_visit in range(0, 2):
            if city in exclude:
                continue
            prob = {"city": city,
                    "drone_visit": drone_visit,
                    "history": pheromone[last_city_visited_by_truck][city] ** c_hist,
                    "distance": ACO_distance(last_city_visited_by_truck, city, last_drone_visit, drone_visit,
                                             drone_distance, truck_distance, last_city_visited_by_drone,
                                             cities_involved)
                    }
            prob["heuristic"] = (1.0 / prob["distance"]) ** c_heur
            prob["prob"] = prob["history"] * prob["heuristic"]
            choices.append(prob)
    return choices


def ACO_prob_select(choices):
    sum_all = 0
    for choice in choices:
        sum_all += choice["prob"]
    if sum_all == 0:
        return choices[random.randint(1, len(choices) - 1)]

    v = random.random()
    for index, choice in enumerate(choices):
        v -= choice["prob"] / sum_all
        if v <= 0.0:
            return choice

    return choices[-1]


def ACO_greedy_select(choices):
    return max(choices, key=lambda e: e["prob"])


def ACO_stepwise_const(phero, c_heur, c_greed):
    perm = [0]
    drone_visits = [0]
    drone_distance = 0
    truck_distance = 0
    last_drone_visit = 0
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    cities_involved = 1

    while True:
        choices = ACO_calculate_choices(
            last_city_visited_by_truck=last_city_visited_by_truck,
            last_drone_visit=last_drone_visit,
            last_city_visited_by_drone=last_city_visited_by_drone,
            drone_distance=drone_distance,
            truck_distance=truck_distance,
            cities_involved=cities_involved,
            exclude=perm,
            pheromone=phero,
            c_heur=c_heur,
            c_hist=1.0
        )

        greedy = random.random() <= c_greed
        if greedy:
            next_choice = ACO_greedy_select(choices)
        else:
            next_choice = ACO_prob_select(choices)

        if next_choice["drone_visit"] == last_drone_visit:
            drone_distance = 0
            truck_distance = 0
            last_city_visited_by_truck = next_choice["city"]
            last_city_visited_by_drone = next_choice["city"],
            last_drone_visit = 0
            cities_involved = 1
        elif next_choice["drone_visit"]:
            drone_distance = ACO_dist(last_city_visited_by_drone, next_choice["city"])
            last_drone_visit = 1
            last_city_visited_by_drone = next_choice["city"]
            cities_involved += 1
        else:
            truck_distance += ACO_dist(last_city_visited_by_truck, next_choice["city"])
            last_city_visited_by_truck = next_choice["city"]
            cities_involved += 1

        perm.append(next_choice["city"])
        drone_visits.append(next_choice["drone_visit"])
        if len(perm) == N_CITIES:
            break
    return perm, drone_visits


def ACO_global_update_pheromone(phero, cand, decay):
    for index, x in enumerate(cand["cities"]):
        if index == len(cand["cities"]) - 1:
            y = cand["cities"][0]
        else:
            y = cand["cities"][index + 1]
        value = ((1.0 - decay) * phero[x][y] + decay * (1.0 / cand["cost"]))
        phero[x][y] = value
        phero[y][x] = value


def ACO_local_update_pheromone(pheromone, cand, c_local_phero, init_phero):
    for index, x in enumerate(cand["cities"]):
        if index == len(cand["cities"]) - 1:
            y = cand["cities"][0]
        else:
            y = cand["cities"][index + 1]
        value = ((1.0 - c_local_phero) * pheromone[x][y]) + (c_local_phero * init_phero)
        pheromone[x][y] = value
        pheromone[y][x] = value


def ACO_search(max_it, num_ants, decay, c_heur, c_local_phero, c_greed):
    best = {"cities": ACO_random_permutation(N_CITIES), "drone_visits": ACO_random_bool_list(N_CITIES)}
    best["cost"] = ACO_cost(best["cities"], best["drone_visits"])
    best_idx = 0
    init_pheromone = 1.0 / (N_CITIES * best["cost"])
    pheromone = ACO_initialise_pheromone_matrix(init_pheromone)

    solutions = [deepcopy(best)]
    bests = []
    global_bests = []
    for iteration in range(max_it):
        current_best = None
        for ant_number in range(num_ants):
            cand = {}
            cand["cities"], cand["drone_visits"] = ACO_stepwise_const(pheromone, c_heur, c_greed)
            cand["cost"] = ACO_cost(cand["cities"], cand["drone_visits"])
            if cand["cost"] < best["cost"]:
                best = cand
                best_idx = iteration
                solutions.append(deepcopy(best))
            if current_best is None or cand["cost"] < current_best:
                current_best = cand["cost"]

            ACO_local_update_pheromone(pheromone, cand, c_local_phero, init_pheromone)
        ACO_global_update_pheromone(pheromone, best, decay)
        # print("Iteration %d: best=%f" % (iteration, best["cost"]))
        c_greed *= 0.99
        bests.append(current_best)
        global_bests.append(best["cost"])
    return best, best_idx, solutions, bests, global_bests


def ACO_main(input_fp, pnum_ants, pdecay, pc_greed):
    if len(sys.argv) == 1:
        sys.argv.append(input_fp)
    else:
        sys.argv[1] = input_fp
    ACO_read_data()

    max_it = int(300 / np.log(N_CITIES))
    num_ants = pnum_ants
    decay = pdecay
    c_heur = 2.5
    c_local_phero = pdecay
    c_greed = pc_greed

    start = time.time()
    best, best_idx, solutions, bests, global_bests = ACO_search(max_it, num_ants, decay, c_heur, c_local_phero, c_greed)
    end = time.time()
    run_time = end - start

    # print(solutions)
    # print("Best solution: c=%f, v=(%s, %s)" % (best["cost"], str(best["cities"]), str(best["drone_visits"])))
    # print('Time to finish run: %ss. Average time per generation: %ss' % (str(run_time), str(run_time / max_it)))
    # plt.plot(range(max_it), bests, 'bo', label='Iteration best')
    # plt.plot(range(max_it), global_bests, 'g.', label='Iteration global best')
    # plt.plot([best_idx], [best["cost"]], 'r+', label='Global best')
    # plt.legend(('Best for each generation', 'Global best'), shadow=True)
    # plt.xlabel('Generation')
    # plt.ylabel('Time taken')
    # plt.show()
    return int(best["cost"]), run_time


def GA_read_data():
    global N_CITIES, CITIES_MATRIX
    N_CITIES = None
    CITIES_MATRIX = None
    if len(sys.argv) == 1:
        print('Specifiy an input file.\ne.g.: %s inputs/example.json' % os.path.basename(sys.argv[0]))
        exit(1)

    input_fp = sys.argv[1]
    with open(input_fp, 'rt') as f:
        input_data = json.load(f)
    N_CITIES = input_data["n"]
    CITIES_MATRIX = np.array(input_data['matrix'])


def GA_tspd_crossover(ind1, ind2):
    tools.cxPartialyMatched(ind1[0], ind2[0])
    tools.cxTwoPoint(ind1[1], ind2[1])
    return ind1, ind2


def GA_tspd_mutation(ind, indpb_cities, indpb_drone_visiting):
    tools.mutShuffleIndexes(ind[0], indpb_cities)
    tools.mutFlipBit(ind[1], indpb_drone_visiting)
    return ind,


def GA_tspd_evaluate(ind):
    is_drone_at_a_customer = False
    last_city_visited_by_truck = 0
    last_city_visited_by_drone = 0
    truck_distance = 0
    drone_distance = 0

    # print('cities:', [city + 1 for city in ind[0]])
    # i = 0

    total_time_taken = 0
    for city, visited_by_drone in zip(ind[0], ind[1]):
        city += 1
        # print('i:', i, ', city:', city)
        # i += 1

        if not visited_by_drone:
            if is_drone_at_a_customer:
                truck_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
            else:
                total_time_taken += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
                # print('totaltime:', total_time_taken)
                # print()
        else:
            if not is_drone_at_a_customer:
                is_drone_at_a_customer = True
                drone_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_drone = city
            else:
                is_drone_at_a_customer = False
                drone_distance += CITIES_MATRIX[last_city_visited_by_drone, city]
                truck_distance += CITIES_MATRIX[last_city_visited_by_truck, city]
                last_city_visited_by_truck = city
                last_city_visited_by_drone = city
                total_time_taken += max(drone_distance, truck_distance)
                # print('totaltime:', total_time_taken)
                # print()
                drone_distance = 0
                truck_distance = 0

    if is_drone_at_a_customer:
        drone_distance += CITIES_MATRIX[last_city_visited_by_drone, 0]
        truck_distance += CITIES_MATRIX[last_city_visited_by_truck, 0]
        total_time_taken += max(drone_distance, truck_distance)
        # print('totaltime:', total_time_taken)
    else:
        total_time_taken += CITIES_MATRIX[last_city_visited_by_truck, 0]

    # print('finaltotaltime:', total_time_taken)
    # exit(1)

    return total_time_taken,


def GA_main(input_fp):
    if len(sys.argv) == 1:
        sys.argv.append(input_fp)
    else:
        sys.argv[1] = input_fp
    global POP_SIZE
    GA_read_data()
    # print('data finished to read')

    POP_SIZE = 200
    N_GENERATIONS = int(3000 / np.log(N_CITIES))
    # js = 19  # statistics justify size: used for space padding

    creator.create("FitnessTSPD", base.Fitness, weights=(-1.0,))
    creator.create("Individual", tuple, fitness=creator.FitnessTSPD)

    toolbox = base.Toolbox()
    # individual and population
    toolbox.register("aux_indices", random.sample, range(0, N_CITIES - 1), N_CITIES - 1)
    toolbox.register("attr_cities", tools.initIterate, list, toolbox.aux_indices)

    toolbox.register("aux_bool", lambda x, y: 0 if random.randint(x, y) <= 50 else 1, 1, 100)
    toolbox.register("attr_drone_visiting", tools.initRepeat, list, toolbox.aux_bool, n=N_CITIES - 1)

    toolbox.register("individual",
                     tools.initCycle,
                     creator.Individual,
                     (toolbox.attr_cities, toolbox.attr_drone_visiting),
                     n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # operators
    toolbox.register("evaluate", GA_tspd_evaluate)
    toolbox.register("mate", GA_tspd_crossover)
    toolbox.register("mutate", GA_tspd_mutation, indpb_cities=0.01, indpb_drone_visiting=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # run the algorithm
    start = time.time()
    population = toolbox.population(n=POP_SIZE)
    # print('%s %s %s %s %s' % ('gen'.ljust(js), 'avg'.ljust(js), 'std'.ljust(js), 'min'.ljust(js), 'max'.ljust(js)))
    bests = []
    best = np.sum(CITIES_MATRIX)
    for gen in range(N_GENERATIONS):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.2, mutpb=0.05)
        fits = toolbox.map(toolbox.evaluate, offspring)
        fits_list = []
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            fits_list.append(fit)
        population = toolbox.select(offspring, k=POP_SIZE)

        # print statistics
        # pavg = np.mean(fits_list)
        # pstd = np.std(fits_list)
        pmin = np.min(fits_list)
        # pmax = np.max(fits_list)
        # print('%s %s %s %s %s' % (str(gen).ljust(js),
        #                           str(pavg).ljust(js),
        #                           str(pstd).ljust(js),
        #                           str(pmin).ljust(js),
        #                           str(pmax).ljust(js)))
        bests.append(pmin)
        if bests[-1] < best:
            best = bests[-1]
            # best_idx = gen
            # best_ind = offspring[np.argmin(fits_list)]
    # print('%s %s %s %s %s' % ('gen'.ljust(js), 'avg'.ljust(js), 'std'.ljust(js), 'min'.ljust(js), 'max'.ljust(js)))

    # plot statistics
    end = time.time()
    run_time = end - start
    # print('Best solution: c=%s, v=%s' % (best, best_ind))
    # print('Time to finish run: %ss.Average time per generation: %ss' % (str(run_time), str(run_time / N_GENERATIONS)))
    # plt.plot(range(N_GENERATIONS), bests, 'bo', [best_idx], [best], 'r+')
    # plt.legend(('Best for each generation', 'Global best'), shadow=True)
    # plt.xlabel('Generation')
    # plt.ylabel('Time taken')
    # plt.show()
    return best, run_time


if __name__ == '__main__':
    verysmall_dataset = 'inputs\\n9_seed2563.json'
    small_dataset = 'inputs\\n17_seed2390.json'
    medium_dataset = 'inputs\\n42_seed8317.json'
    large_dataset = 'inputs\\n60_seed5049.json'
    verylarge_dataset = 'inputs\\n563_seed5842.json'

    # data_sets = [verysmall_dataset, small_dataset, medium_dataset, large_dataset, verylarge_dataset]
    data_sets = OrderedDict([(small_dataset, 17), (large_dataset, 60)])
    # with open('results\\results_GA_%s.csv' % os.path.basename(data_set), 'wt') as f:
    #     f.write('run_number;cost;run_time\n')
    #     costs = []
    #     runtimes = []
    #     for i in range(10):
    #         cost, runtime = GA_main(data_set)
    #         costs.append(cost)
    #         runtimes.append(runtime)
    #         print('Finished run %d/10 with GA, dataset %s' % (i + 1, data_set))
    #         f.write('%d;%s;%0.2f\n' % (i + 1, str(cost), runtime))
    #     f.write('average;%0.2f;%0.2f\n' % (np.mean(costs), np.mean(runtimes)))
    #     f.write('minimum;%0.2f;%0.2f\n' % (np.min(costs), np.min(runtimes)))
    #     f.write('maximum;%0.2f;%0.2f\n' % (np.max(costs), np.max(runtimes)))
    #     f.write('stddev;%0.2f;%0.2f\n' % (np.std(costs), np.std(runtimes)))
    configs = [
        # {
        #     "num_ants": 10,
        #     "decay": 0.1,
        #     "c_greed": 0.7
        # },
        # {
        #     "num_ants": 100,
        #     "decay": 0.1,
        #     "c_greed": 0.7
        # },
        # {
        #     "num_ants": 2,
        #     "decay": 0.1,
        #     "c_greed": 0.7
        # },
        # {
        #     "num_ants": 10,
        #     "decay": 0.5,
        #     "c_greed": 0.7
        # },
        # {
        #     "num_ants": 10,
        #     "decay": 0.01,
        #     "c_greed": 0.7
        # },
        # {
        #     "num_ants": 10,
        #     "decay": 0.1,
        #     "c_greed": 0.99
        # },
        # {
        #     "num_ants": 10,
        #     "decay": 0.1,
        #     "c_greed": 0.1
        # }
        {
            "num_ants": 100,
            "decay": 0.5,
            "c_greed": 0.99
        },
        {
            "num_ants": 100,
            "decay": 0.01,
            "c_greed": 0.99
        }
    ]
    for data_set, config in zip(data_sets, configs):
        # for config in configs:
        fn = 'results_AC_n%d_a%d_d%0.2f_g%0.2f.csv' % (data_sets[data_set],
                                                       config["num_ants"],
                                                       config["decay"],
                                                       config["c_greed"]
                                                       )
        with open('results\\%s' % fn, 'wt') as f:
            f.write('run_number;cost;run_time\n')
            costs = []
            runtimes = []
            for i in range(10):
                cost, runtime = ACO_main(data_set, config["num_ants"], config["decay"], config["c_greed"])
                costs.append(cost)
                runtimes.append(runtime)
                print('Finished run %d/10 with ACO, dataset %s' % (i + 1, data_set))
                f.write('%d;%s;%0.2f\n' % (i + 1, str(cost), runtime))
            f.write('average;%0.2f;%0.2f\n' % (np.mean(costs), np.mean(runtimes)))
            f.write('minimum;%0.2f;%0.2f\n' % (np.min(costs), np.min(runtimes)))
            f.write('maximum;%0.2f;%0.2f\n' % (np.max(costs), np.max(runtimes)))
            f.write('stddev;%0.2f;%0.2f\n' % (np.std(costs), np.std(runtimes)))
