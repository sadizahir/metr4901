"""
Trainer code.
"""

from __future__ import print_function

import vtk
import random
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids
from helper_patches import create_patch
from constants import LANDMARK_REGIONS

# Set the mesh filename
meshFilename = "bones/Asymknee13_boneSurface.vtk"
landmarksFilename = "bones/Asymknee13.csv"

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[1]

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
landmarkSize = 7
st = time.time()
for i in landmarkIds:
	landmarkPatches.append(create_patch(model, modelGraph, idArray, invIdArray, i, landmarkSize))
t = time.time() - st
print("Generated {} landmark patches of size {} in {} seconds.".format(len(landmarkIds), landmarkSize, t))

# Paint the patches
for patch in landmarkPatches:
	for point in patch:
		scalars.SetValue(point, 2)

# Apply scalars to model
model.GetPointData().SetScalars(scalars)

# Output the model with scalars
outputFilename = subBoneFilename.split(".")[0] + "_landmarks_Size{}".format(landmarkSize) + "." + subBoneFilename.split(".")[1]
writer = vtk.vtkPolyDataWriter()
writer.SetFileName(outputFilename)
writer.SetInputData(model)
writer.Write()