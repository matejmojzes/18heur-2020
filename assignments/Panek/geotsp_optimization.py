from objfun_geotsp import GeoTSP


spots_file = "/Users/ondrej.panek/Documents/GitHub/heur/assignments/Panek/spots.csv"
# spots_file = "spots.csv"

spots = []
with open(spots_file) as file:
    for line in file:
        spots.append(line)

n = len(spots)
spots = list(map(lambda s: s.strip(), spots))  # remove blank spaces
key = open("/Users/ondrej.panek/Documents/School/HEUR/gapikey.txt", "r").read()  # API key

dm_path = "/Users/ondrej.panek/Documents/GitHub/heur/assignments/Panek/distance_matrix.csv"

TSP = GeoTSP(spots, dm_path=dm_path)
"""
import numpy as np
import googlemaps
from datetime import datetime


now = datetime.now()
gmaps = googlemaps.Client(key)
spots_chunks = list(TSP.chunks(24))
matrix = np.zeros([n, n])

print(spots)
print(spots_chunks[0])
mtype = "distance"

for n in range(n):  # For each spot
    abs_spot_number = 0  # Absolute position number of the spot in the row
    for m in range(len(spots_chunks)):  # For each chunk in the spots list
        dm = gmaps.distance_matrix(spots[n], spots_chunks[m], mode="driving", units="metric", departure_time=now)
        for k in range(len(spots_chunks[m])):  # Compute distance for each element in the m-th chunk
            #  needs to move with chunks
            matrix[n][abs_spot_number] = dm["rows"][0]["elements"][k][mtype]['value']
            abs_spot_number += 1

print(matrix)
"""

"""
i = 0
for r in dm["rows"]:
    j = 0
    for dist in r["elements"]:
        print(f"From {TSP.spots[i].strip()}")
        print(f"To {TSP.spots[j]}")
        print(f"Exact distance: {dist['distance']['value']} m")
        print(f"Rounded: {dist['distance']['text']}\n")
        j += 1
    i += 1
"""