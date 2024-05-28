import random
import time
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, namedtuple
from typing      import Set, List, Tuple, Iterable, Dict
import os

import numpy as np
from networkx.readwrite import json_graph
import json
import networkx as nx


City   = complex   # e.g. City(300, 100)\
Cities = frozenset # A set of cities
Tour   = list      # A list of cities visited, in order
TSP    = callable  # A TSP algorithm is a callable function
Link   = Tuple[City, City] # A city-city link

def distance(A: City, B: City) -> float:
    "Distance between two cities"
    return abs(A - B)
def graph_distance(A: City, B: City, adjacency_matrix ) -> float:
    "Distance between two cities"
    return adjacency_matrix[A][B]
def shortest(tours: Iterable[Tour]) -> Tour:
    "The tour with the smallest tour length."
    return min(tours, key=tour_length)

def tour_length(tour: Tour, adjacency_matrix) -> float:
    "The total distances of each link in the tour, including the link from last back to first."
    return sum(graph_distance(tour[i], tour[i - 1], adjacency_matrix) for i in range(len(tour)))

def valid_tour(tour: Tour, cities: Cities) -> bool:
    "Does `tour` visit every city in `cities` exactly once?"
    return Counter(tour) == Counter(cities)

def random_cities(n, seed=1234, width=9999, height=6666) -> Cities:
    "Make a set of n cities, sampled uniformly from a (width x height) rectangle."
    random.seed((n, seed)) # To make `random_cities` reproducible
    return Cities(City(random.randrange(width), random.randrange(height))
                  for c in range(n))


Segment = list  # A portion of a tour; it does not loop back to the start.


def plot_tour(tour: Tour, style='bo-', hilite='rs', title=''):
    "Plot every city and link in the tour, and highlight the start city."
    scale = 1 + len(tour) ** 0.5 // 10
    plt.figure(figsize=((3 * scale, 2 * scale)))
    start = tour[0]
    plot_segment([*tour, start], style)
    plot_segment([start], hilite)
    plt.title(title)


def Xs(cities) -> List[float]: "X coordinates"; return [c.real for c in cities]


def Ys(cities) -> List[float]: "Y coordinates"; return [c.imag for c in cities]


def plot_segment(segment: Segment, style='bo:'):
    "Plot every city and link in the segment."
    plt.plot(Xs(segment), Ys(segment), style, linewidth=2 / 3, markersize=4, clip_on=False)
    plt.axis('scaled');
    plt.axis('off')

def run(tsp: callable, cities: Cities, adjacency_matrix: np.ndarray):
    """Run a TSP algorithm on a set of cities and plot/print results."""
    t0   = time.perf_counter()
    tour = tsp(cities, adjacency_matrix)
    t1   = time.perf_counter()
    L    = tour_length(tour, adjacency_matrix)
    print(f"length {round(L):,d} tour of {len(cities)} cities in {t1 - t0:.3f} secs")
    return L
def nearest_tsp(cities, adjacency_matrix, start=None) -> Tour:
    """Create a partial tour that initially is just the start city.
    At each step extend the partial tour to the nearest unvisited neighbor
    of the last city in the partial tour, while there are unvisited cities remaining."""
    start = start or first(cities)
    tour = [start]
    unvisited = set(cities) - {start}
    def extend_to(C): tour.append(C); unvisited.remove(C)
    while unvisited:
        extend_to(nearest_neighbor(tour[-1], unvisited, adjacency_matrix))
    return tour
def first(collection):
    """The first element of a collection."""
    return next(iter(collection))

def nearest_neighbor(A: City, cities, adjacency_matrix) -> City:
    """Find the city C in cities that is nearest to city A."""
    return min(cities, key=lambda C: graph_distance(C, A, adjacency_matrix))



filename = "./skewed_w_dist_gi_4_12.json"
if os.path.exists(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

for N in [4, 5, 6, 8, 10, 12]:
    for s in np.arange(-2.0, 2.0, 0.5):
        s = np.round(s, 1)

        adj_mat = data["N_{}".format(N)]["s_{}".format(s)]
        for it in range(0, 101):

            graph = json_graph.adjacency_graph(adj_mat["it_{}".format(it)])

            adjacency_matrix = nx.to_numpy_array(graph)
            nodelist = frozenset(graph.nodes())
            json_dic = {}

            tsp_results = run(nearest_tsp, nodelist, adjacency_matrix)
            with open("./NN_test_results/N_{}_s_{}_it_{}_NN_test_results.json".format(N, s, it), "w") as outfile:
                json.dump(tsp_results, outfile)

#plt.show()