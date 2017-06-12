"""
Paints landmarks onto a mesh, but larger than a single point.
Sample code to show how to use the "create_patch" method.
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
meshFilename = "bones/Asymknee11_boneSurface.vtk"
landmarksFilename = "bones/Asymknee11.csv"

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

# These scalars may be set to many things, just initialising here
scalars = vtk.vtkDoubleArray()
scalars.SetNumberOfValues(model.GetNumberOfPoints())

# For example, set all scalars to 1
for i in range(model.GetNumberOfPoints()):
	scalars.SetValue(i, 1)

# Make some patches at the landmarks, of a certain size
landmarkPatches = []
landmarkSize = 0
st = time.time()

# Iterative way
for i in landmarkIds:
	landmarkPatches.append(create_patch(model, modelGraph, idArray, invIdArray, i, landmarkSize))

t = time.time() - st

print("Generated {} landmark patches of size {} in {} seconds.".format(len(landmarkIds), landmarkSize, t))
print("Average patch generation time: {} seconds.".format(t/len(landmarkIds)))
print(model.GetNumberOfPoints())

# Paint the patches
for i, patch in enumerate(landmarkPatches):
	actualLandmarkNo = LANDMARK_REGIONS[subBone][i]

	for point in patch:
		scalars.SetValue(point, 2)

	# Apply scalars to model
	model.GetPointData().SetScalars(scalars)

	# Output the model with scalars
	if landmarkSize == 0:
		landmarkSizeName = ""
	else:
		landmarkSizeName = "_Size{}".format(landmarkSize)
	outputFilename = subBoneFilename.split(".")[0] + "_point_landmark{}{}".format(actualLandmarkNo, landmarkSizeName) + "." + subBoneFilename.split(".")[1]
	writer = vtk.vtkPolyDataWriter()
	writer.SetFileName(outputFilename)
	writer.SetInputData(model)
	writer.Write()

	scalars = vtk.vtkDoubleArray()
	scalars.SetNumberOfValues(model.GetNumberOfPoints())

	for i in range(model.GetNumberOfPoints()):
		scalars.SetValue(i, 1)