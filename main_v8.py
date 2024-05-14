import random
import geopy.distance
from deap import base, creator, tools, algorithms
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
import time
from scipy.spatial import KDTree
import os
import requests
import functools
from tenacity import retry, stop_after_attempt, wait_exponential

# Create a session object for HTTP requests
session = requests.Session()
# Create ThreadPool for concurrent requests
pool = ThreadPool()

st = time.time()
#balance dominant
BD_CROSS = 0.70
BD_MUTATE = 0.10
#crossover dominant
CD_CROSS = 0.80
CD_MUTATE = 0.01
#mutation dominant
MD_CROSS = 0.60
MD_MUTATE = 0.40

DEFAULT_SPEED = 80  # km/h
NUM_SITES = 6  # Number of rendezvous sites in each individual
N_GEN = 50  # Number of generations
N_POP = 20  # Number of individuals in each generation
P_CROSS = MD_CROSS  # crossover probaility
P_MUTATE = MD_MUTATE  # mutation probability
G_MUTATE = 0.2  # gene mutation rate

print("Running:")
print(f"Number of site {NUM_SITES}")
print(f"Mutation {P_MUTATE}")
print(f"Crossover {P_CROSS}")

CURR_SITE = 'bangkok'
OUTPUT_PATH = f'output/{CURR_SITE}/optimize/GEN{N_GEN}/RS{NUM_SITES}/CROSS{P_CROSS}_MUTATE{P_MUTATE}/'
#OUTPUT_PATH = f'output/{CURR_SITE}/test/' #for testing
DATA_PATH = f'data/{CURR_SITE}/optimize'
#DATA_PATH = f'data/{CURR_SITE}/test_data' #for testing
# save best result to csv
# FOLDER_PATH = f"results_{NUM_SITES}"
# create dir if not exist
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
BEST_RESULT_PATH = f'{OUTPUT_PATH}/best_rps.csv'

print(f"Site: {CURR_SITE}")

# Read your latitude and longitude data from a CSV file (replace 'Long_Lats.csv' with your file)
df = pd.read_csv("./GA/data/bangkok/optimize/patients.csv", delimiter=',', skiprows=0, low_memory=False)
dfRe = pd.read_csv("./GA/data/bangkok/optimize/rendezvous.csv", delimiter=',', skiprows=0, low_memory=False)
msus = pd.read_csv('./GA/data/bangkok/optimize/msus.csv', delimiter=',', skiprows=0, low_memory=False)

# Load data
# [latitude, longitude, count]
patients = pd.read_csv(f'{DATA_PATH}/patients.csv').to_numpy()
#random.seed(42)
# patients = np.array(random.sample(list(patients), 100))
msus = pd.read_csv(f'{DATA_PATH}/msus.csv').to_numpy()
rendezvous_sites_raw = pd.read_csv(f'{DATA_PATH}/rendezvous.csv')
rendezvous_sites = rendezvous_sites_raw.to_numpy()

msu_kdtree = KDTree(msus[:, 1:3])
rps_coords = rendezvous_sites[:, 1:3]

"""
# Convert speed columns to integers and filter sites
rendezvous_sites['download_speed'] = pd.to_numeric(rendezvous_sites['download_speed'], errors='coerce').fillna(0)
rendezvous_sites['upload_speed'] = pd.to_numeric(rendezvous_sites['upload_speed'], errors='coerce').fillna(0)
rendezvous_sites = rendezvous_sites[
    (rendezvous_sites['download_speed'] >= 25) &
    (rendezvous_sites['upload_speed'] >= 3)
    ].to_numpy()

rps_coords = rendezvous_sites[:, 1:3]
"""


def calculate_rps_weight(rps_coords, patients_coords):
    kdtree_patients = KDTree([x[:2] for x in patients_coords])
    # for each rps, find 10 closest patients, take sum of their count
    rps_weight = []
    for rps in rps_coords:
        _, idx = kdtree_patients.query([rps], k=10)
        rps_weight.append(sum([patients_coords[i][2] for i in idx[0]]))
    return rps_weight


rps_weight = calculate_rps_weight(rps_coords, patients)


# Calculate distances between coordinates
# def calculate_distance(coord1, coord2):
#   return geopy.distance.geodesic(coord1, coord2).kilometers


# Define a fallback function to return a high travel time in case of retries exhaustion
def retry_fallback(retry_state):
    # You can log the failure or perform other actions here
    print("Retries exhausted. Returning a fallback travel time.")
    return 1000  # Fallback travel time in minutes


# Calculate travel time based on distance and speed assumption
def calculate_travel_time(c1, c2):
    return cached_travel_time(tuple(c1), tuple(c2))


@functools.lru_cache(maxsize=4096)  # cache up to 4096 unique routes (can be adjusted as needed)
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=30),
       retry_error_callback=retry_fallback)  # Retry up to 10 times
def cached_travel_time(c1, c2):
    # OSRM API endpoint for routing
    osrm_endpoint = "http://139.59.32.22:5000/route/v1/driving/"
    # Format coordinates for the API request
    coordinates = f"{c1[1]},{c1[0]};{c2[1]},{c2[0]}"
    # Construct the full URL for the API request
    url = osrm_endpoint + coordinates + "?overview=false"

    # Use the global session object for making the request
    response = session.get(url, timeout=10)  # Adding a timeout for good measure
    data = response.json()

    # Check if the response contains valid routing data
    if data['code'] == 'Ok':
        # Extract travel time from the first (and only) route
        travel_time_seconds = data['routes'][0]['duration']
        # Convert travel time to minutes
        travel_time_minutes = travel_time_seconds / 60
        return travel_time_minutes
    else:
        # Return a high value or handle the error appropriately if no route is found
        return 1000  # Example fallback value


open(f'{OUTPUT_PATH}/log.txt', 'w').close()


# Define fitness function
def fitness_function(selected_sites):
    # get index of selected sites
    selected_sites_index = [i for i in range(len(selected_sites[0])) if selected_sites[0][i] == 1]
    coords = [rps_coords[i] for i in selected_sites_index]
    kdtree = KDTree(coords)
    total_ambulance_time = 0
    total_msu_time = 0
    request_time_counter = 0
    request_time_avg = 0
    request_time_total = 0
    st = time.time()
    for patient in patients:
        request_st = time.time()
        _, nearest_rps_id = kdtree.query(patient[:2])
        _, idx_2 = msu_kdtree.query(patient[:2])
        nearest_mus_of_patient = msus[idx_2, 1:3]
        patient_count = patient[2]
        min_travel_time = calculate_travel_time(patient[:2], rps_coords[selected_sites_index[nearest_rps_id]])
        msu_travel_time = calculate_travel_time(patient[:2], nearest_mus_of_patient)
        if msu_travel_time < min_travel_time:
            total_ambulance_time += ((1 + msu_travel_time) ** 2) * patient_count
        else:
            total_ambulance_time += ((1 + min_travel_time) ** 2) * patient_count

            # msu to rps
            _, idx = msu_kdtree.query([rps_coords[selected_sites_index[nearest_rps_id]]])
            travel_time = calculate_travel_time(rps_coords[selected_sites_index[nearest_rps_id]], msus[idx[0], 1:3])
            if travel_time > 30:
                travel_time = 1000
            # print(travel_time, patient_count)
            total_msu_time += ((1 + travel_time) ** 2) * patient_count
        request_et = time.time()
        request_time_counter += 1
        request_time = request_et - request_st
        request_time_total = request_time_total + request_time
    et = time.time()
    #calculation time per fitness
    fitness_time = et - st
    fitness_time = fitness_time / 60
    print(f"Fitness Calculation time: {fitness_time:.3f} minutes")

    #average request time per fitness
    request_time_avg = request_time_total / request_time_counter
    print(f"Request time average: {request_time_avg:.3f} seconds")

    print(f"Fitness done {total_ambulance_time + total_msu_time}", file=open(f'{OUTPUT_PATH}/log.txt', 'a'))
    # print(total_ambulance_time)
    # print(total_msu_time)
    return total_ambulance_time + total_msu_time


def generate_individual():
    sites = []
    while len(sites) < NUM_SITES:
        # sites = random.sample(range(len(rendezvous_sites)), NUM_SITES)
        site = random.choices(range(len(rendezvous_sites)), weights=rps_weight, k=1)[0]
        if site not in sites:
            sites.append(site)
    individual = [0] * len(rendezvous_sites)
    for site in sites:
        individual[site] = 1
    return individual


def mate(ind1, ind2):
    child_1 = toolbox.individual()
    child_2 = toolbox.individual()

    # Initialize children with zeros
    child_1[0] = [0] * len(ind1[0])
    child_2[0] = [0] * len(ind1[0])

    # print("sum", sum(ind1[0]), sum(ind2[0]))
    count_1 = count_2 = 0
    split = []
    for i in range(len(ind1[0])):
        if ind1[0][i] == 1 and ind2[0][i] == 1:
            # Both parents have selected the site
            child_1[0][i] = 1
            child_2[0][i] = 1
            count_1 += 1
            count_2 += 1
        elif ind1[0][i] != ind2[0][i]:
            split.append(i)

    for i in split:
        if count_1 >= NUM_SITES:
            pick = 2
        elif count_2 >= NUM_SITES:
            pick = 1
        elif random.random() < 0.5: #this is crossOver rate
            pick = 1
        else:
            pick = 2

        # Only one parent has selected the site
        if pick == 1:
            child_1[0][i] = 1
            count_1 += 1
        elif pick == 2:
            child_2[0][i] = 1
            count_2 += 1

    n1 = sum([1 for i in child_1[0] if i == 1])
    n2 = sum([1 for i in child_2[0] if i == 1])
    # print(n1, n2)
    assert n1 == n2
    return child_1, child_2


# Genetic Algorithm Configuration
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize both distance and time
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("map", pool.map)
toolbox.register("attr_bool", generate_individual)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", mate)


# Mutation operator: Change one randomly selected site
def mutate(individual):
    selected_indices = [i for i, val in enumerate(individual[0]) if val == 1]
    # gene mutate
    n_mutate = G_MUTATE * len(selected_indices)
    not_selected_indices = [i for i, val in enumerate(individual[0]) if val == 0]
    not_selected_weights = [rps_weight[i] for i in not_selected_indices]
    # random sample n_mutate indices to mutate
    removed_indices = random.sample(selected_indices, int(n_mutate))
    added_indices = []
    while len(added_indices) < min(int(n_mutate), len(not_selected_indices)):
        idx = random.choices(not_selected_indices, weights=not_selected_weights, k=1)[0]
        if idx not in added_indices:
            added_indices.append(idx)

    for i in removed_indices:
        individual[0][i] = 0
    for i in added_indices:
        individual[0][i] = 1

    n = sum([1 for i in individual[0] if i == 1])
    assert n == NUM_SITES
    return individual,


# Register Tournament Selection
TOURNAMENT_SIZE = 3  # Adjust this as needed

toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# add max, min, avg
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)
stats.register("min", min)
stats.register("avg", np.mean)


def visiualize(pop):
    # visualize patients and rps
    best_pop = pop[0][0]
    best_rps = [rendezvous_sites[i] for i in range(len(best_pop)) if best_pop[i] == 1]
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    all_patients = pd.read_csv(f'{DATA_PATH}/patients.csv').to_numpy()
    plt.scatter(all_patients[:, 2], all_patients[:, 1], c='b', s=0.01)
    plt.scatter([x[2] for x in best_rps], [x[1] for x in best_rps], c='r', s=1)
    plt.scatter(msus[:, 2], msus[:, 1], c='g', s=50)

    plt.savefig(f'{OUTPUT_PATH}/result.png')
    # plt.show()


def handle_result(pop, gen):
    PATIENTS_TO_RPS_PAIR = f'{OUTPUT_PATH}/patients_to_rps{gen}.csv'
    RPS_TO_MSU_PAIR = f'{OUTPUT_PATH}/rps_to_msu{gen}.csv'
    best_pop = pop[0][0]
    # print(rendezvous_sites[:10])
    best_rps = [rendezvous_sites[i] for i in range(len(best_pop)) if best_pop[i] == 1]
    best_rps_df = pd.DataFrame(best_rps, columns=rendezvous_sites_raw.columns)
    best_rps_df['site_id'] = best_rps_df['site_id'].astype(int)
    best_rps_df.to_csv(BEST_RESULT_PATH, index=False)

    test_individual = toolbox.individual()
    test_individual[0] = best_pop

    kdtree_rps = KDTree([x[1:3] for x in best_rps])
    kdtree_msu = KDTree([x[1:3] for x in msus])
    all_patients = pd.read_csv(f'{DATA_PATH}/patients.csv').to_numpy()
    patiens_time = []
    patients_to_rps = []
    rps_to_msu = []
    rps_time = []
    rps_count = [0 for _ in range(len(best_rps))]
    for patient in all_patients:
        if int(patient[0]) % 50000 == 0:
            print(int(patient[0]))
        _, idx = kdtree_rps.query([patient[1:3]])
        nearest_rps_idx = idx[0]
        rps_count[nearest_rps_idx] += 1
        travel_time = calculate_travel_time(patient[1:3], best_rps[nearest_rps_idx][1:3])
        patiens_time.append(travel_time)
        patients_to_rps.append([patient[0], best_rps[nearest_rps_idx][0], travel_time])

    for rps_idx, rps in enumerate(best_rps):
        _, idx = kdtree_msu.query([rps[1:3]])
        nearest_msu_idx = idx[0]
        travel_time = calculate_travel_time(rps[1:3], msus[nearest_msu_idx][1:3])
        for _ in range(rps_count[rps_idx]):
            rps_time.append(travel_time)
        rps_to_msu.append([rps[0], msus[idx][0], travel_time])

    df = pd.DataFrame(patients_to_rps, columns=['patient_id', 'rps_id', 'travel_time'])
    df.to_csv(PATIENTS_TO_RPS_PAIR, index=False)
    df = pd.DataFrame(rps_to_msu, columns=['rps_id', 'msu_id', 'travel_time'])
    df.to_csv(RPS_TO_MSU_PAIR, index=False)

    # print max, min, avg time of patients to rps
    print("Patients to RPS:")
    print(f"Max: {round(max(patiens_time), 4)} min")
    print(f"Min: {round(min(patiens_time), 4)} min")
    print(f"Avg: {round(np.mean(patiens_time), 4)} min")
    print(f"Total: {round(sum(patiens_time), 4)} min")

    # print max, min, avg time of rps to msu
    print("RPS to MSU:")
    print(f"Max: {round(max(rps_time), 4)} min")
    print(f"Min: {round(min(rps_time), 4)} min")
    print(f"Avg: {round(np.mean(rps_time), 4)} min")
    print(f"Total: {round(sum(rps_time), 4)} min")


# Main function
def main():
    if os.path.exists(BEST_RESULT_PATH):
        pop = toolbox.population(n=N_POP)
        # create new individual
        best_rps = pd.read_csv(BEST_RESULT_PATH)
        selected_sites_id = best_rps["site_id"].to_list()

        individual = toolbox.individual()
        individual[0] = [0] * len(rendezvous_sites)
        for selected_site_id in selected_sites_id:
            for i in range(len(rendezvous_sites)):
                if int(rendezvous_sites[i][0]) == int(selected_site_id):
                    individual[0][i] = 1

        n = sum([1 for i in individual[0] if i == 1])
        assert n == NUM_SITES
        pop.append(individual)
    else:
        pop = toolbox.population(n=N_POP)

    for gen in range(1, N_GEN + 1):
        # This is the GA operation that evolves the population
        algorithms.eaMuPlusLambda(pop, toolbox, mu=N_POP, lambda_=N_POP, cxpb=P_CROSS, mutpb=P_MUTATE, ngen=1,
                                  verbose=True,
                                  stats=stats)
        print(f"Gen {gen}")
        # Check if current generation is a multiple of 5
        #if gen % 5 == 0 or gen == N_GEN:
        print(f"Saving output after generation {gen}")
            # Save the current state or result. Adjust the function as needed.
        handle_result(pop, gen)
        # Calculate the percentage of completion
        # completion_percentage = (gen / N_GEN) * 100
        # print(f"Completion: {completion_percentage:.2f}% after generation {gen}/{N_GEN}")

    print("Fitness Values:")
    print(toolbox.evaluate(pop[0]))

    # save best pop to csv
    handle_result(pop, gen)

    visiualize(pop)

    et = time.time()
    duration_seconds = et - st
    duration_minutes = duration_seconds / 60
    print(f"Time: {duration_minutes:.2f} minutes")
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
