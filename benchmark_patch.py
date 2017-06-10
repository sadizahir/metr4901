"""
Benchmarks the create_patch method.
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
from helper_patches import create_patch_optimised
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

for size in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
	i = 1000
	st = time.time()
	patch = create_patch_optimised(model, modelGraph, idArray, invIdArray, i, size)
	t = time.time() - st
	print("Took {} seconds to create dijkstra patch of size {}, with {} vertices".format(t, size, len(patch)))

# Iterative way
# for i in landmarkIds:
# 	landmarkPatches.append(create_patch(model, modelGraph, idArray, invIdArray, i, landmarkSize))

# Parallel way
# Parallel(n_jobs=2)(delayed(create_patch)(model, modelGraph, idArray, invIdArray, i, landmarkSize) for i in landmarkIds)