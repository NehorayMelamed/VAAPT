import numpy as np
from shapely.geometry import Point
import networkx as nx
from shapely.strtree import STRtree

import time

# def get_GCC(arr):
#
#     n = len(arr)
#     circles = [Point(x[0], x[1]).buffer(x[2]) for x in arr]
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#
#     for i, circle_i in enumerate(circles):
#         for j, circle_j in enumerate(circles):
#             if circle_i.intersects(circle_j):
#                 G.add_edge(i,j)

#
#     return list(nx.connected_components(G))

def get_GCC(arr):

    n = len(arr)
    circles = [Point(x[0], x[1]).buffer(x[2]) for x in arr]
    s = STRtree(circles)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, circle_i in enumerate(circles):
        result = s.query(circle_i)
        for j, circle_j in enumerate(circles):
            if circle_j in result:
                G.add_edge(i, j)
    return list(nx.connected_components(G))


n = 100
arr = np.random.rand(n,3)
start = time.time()
GCC = get_GCC(arr)
end = time.time()
print(end - start)

# 1000 16.52084231376648
# 600 5.982313871383667

