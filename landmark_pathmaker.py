"""
Paints (Dijkstra) distance heatmaps to every landmark on a mesh and saves
each one.
"""

from __future__ import print_function

import vtk
import random
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids
from helper_patches import create_patch
from constants import LANDMARK_REGIONS

# Set the mesh filename
meshFilename = "bones/Asymknee22_boneSurface.vtk"
landmarksFilename = "bones/Asymknee22.csv"

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[2]

# Check to see if it exists
subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
open(subBoneFilename, 'r')

# Load the mesh model and generate graph
model = read_mesh(subBoneFilename)
modelGraph, idArray, invIdArray = generate_graph(model)

# Load all the landmarks Point IDs for this mesh
landmarks = load_landmarks(landmarksFilename)
allLandmarkIds = get_landmark_ids(model, landmarks)
landmarkIds = []
for i, ID in enumerate(allLandmarkIds):
	if i in LANDMARK_REGIONS[subBone]:
		landmarkIds.append(ID)

# Go through all landmarks and make Dijkstra heatmaps for them
scalars = vtk.vtkDoubleArray()
scalars.SetNumberOfValues(model.GetNumberOfPoints())
writer = vtk.vtkPolyDataWriter()

for i, ID in enumerate(landmarkIds):
	actualLandmarkNo = LANDMARK_REGIONS[subBone][i]

	# Perform the dijkstra algorithm
	dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
	dijkstra.SetInputData(model)
	dijkstra.SetStartVertex(ID)
	dijkstra.SetEndVertex(0) # just a random vertex so that it will compute all the paths
	dijkstra.Update()
	dijkstra.GetCumulativeWeights(scalars)

	# Apply scalars to model
	model.GetPointData().SetScalars(scalars)

	# Output the model with scalars
	outputFilename = subBoneFilename.split(".")[0] + "_heatmap_landmark{}".format(actualLandmarkNo) + "." + subBoneFilename.split(".")[1]
	writer.SetFileName(outputFilename)
	writer.SetInputData(model)
	writer.Write()