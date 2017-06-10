"""
Landmark Locator

Determines which bone each landmark belongs to.
Only works on meshes that have been processed by "label unconnected regions"
in SMILI.
"""

from __future__ import print_function

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids

# Set the mesh filename
meshFilename = "bones/Asymknee11_boneSurfaceRegions.vtk"
landmarksFilename = "bones/Asymknee11.csv"

# Check to see if it exists
open(meshFilename, 'r')

# Load the mesh model and generate graph
model = read_mesh(meshFilename)
modelGraph, idArray, invIdArray = generate_graph(model)

# Load all the landmarks Point IDs for this mesh
landmarks = load_landmarks(landmarksFilename)
landmarkIds = get_landmark_ids(model, landmarks)

# Find the landmarks and their associated scalar values
landmarkLocations = {}
scalars = model.GetPointData().GetScalars()

for i in landmarkIds:
	location = scalars.GetValue(i)
	if location not in landmarkLocations.keys():
		landmarkLocations[location] = []
	landmarkLocations[location].append(i)

for location in landmarkLocations.keys():
	print("Location: {}".format(location))
	for i in landmarkLocations[location]:
		print("Landmark #{} at ID {}".format(landmarkIds.index(i), i))
	print("")