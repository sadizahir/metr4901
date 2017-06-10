"""
Paints Landmarks as scalar = 2 dots on a mesh.
"""

from __future__ import print_function

import vtk
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids
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

# These scalars may be set to many things, just initialising here
scalars = vtk.vtkDoubleArray()
scalars.SetNumberOfValues(model.GetNumberOfPoints())

# For example, set all scalars to 1
for i in range(model.GetNumberOfPoints()):
	scalars.SetValue(i, 1)

# And then highlight all the landmarks
for i in landmarkIds:
	scalars.SetValue(i, 2)

# Apply scalars to model
model.GetPointData().SetScalars(scalars)

# Output the model with scalars
outputFilename = subBoneFilename.split(".")[0] + "_landmarks" + "." + subBoneFilename.split(".")[1]
writer = vtk.vtkPolyDataWriter()
writer.SetFileName(outputFilename)
writer.SetInputData(model)
writer.Write()