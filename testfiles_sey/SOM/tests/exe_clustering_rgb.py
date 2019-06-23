# Reference Page of SOM clustering exaples
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/examples/som_examples.py
#  : https://github.com/annoviko/pyclustering/blob/master/pyclustering/nnet/som.py
# Reference Image
#  : https://github.com/annoviko/pyclustering/blob/master/docs/img/target_som_processing.png


import random
from pyclustering.utils import read_sample
from pyclustering.nnet.som import som, type_conn, type_init, som_parameters
from pyclustering.samples.definitions import FCPS_SAMPLES


# read sample 'Lsun' from file
sample = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
# [[2.0, 3.0], [0.387577, 0.268546], [0.17678, 0.582963], [3.277701, 0.814082], [0.387577, 0.17678], ...]

# create SOM parameters
parameters = som_parameters()

# create self-organized feature map with size 7x7
rows = 15  # five rows
cols = 15  # five columns
structure = type_conn.grid_four;  # each neuron has max. four neighbors.
network = som(rows, cols, structure, parameters)

# train network on 'Lsun' sample during 100 epouchs.
network.train(sample, 100)

# simulate trained network using randomly modified point from input dataset.
index_point = random.randint(0, len(sample) - 1)
point = sample[index_point]  # obtain randomly point from data
point[0] += random.random() * 0.2  # change randomly X-coordinate
point[1] += random.random() * 0.2  # change randomly Y-coordinate
index_winner = network.simulate(point)

# check what are objects from input data are much close to randomly modified.
index_similar_objects = network.capture_objects[index_winner]

# neuron contains information of encoded objects
print("Point '%s' is similar to objects with indexes '%s'." % (str(point), str(index_similar_objects)))
print("Coordinates of similar objects:")
for index in index_similar_objects: print("\tPoint:", sample[index])

# result visualization:
# show distance matrix (U-matrix).
network.show_distance_matrix()
# show density matrix (P-matrix).
network.show_density_matrix()
# show winner matrix.
network.show_winner_matrix()
# show self-organized map.
network.show_network()