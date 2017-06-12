"""
Trainer code.
"""

from __future__ import print_function

import os
import vtk
import numpy as np
import time
import joblib
from sklearn.ensemble import RandomForestRegressor

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids
from helper_patches import create_patch
from helper_patches import create_patch_optimised
from helper_patches import get_random_points
from helper_features import get_features
from constants import LANDMARK_REGIONS

SAMPLE_RATE = 0.2
ORDER = 7
PATCH_SIZE = 14
PATCH_METHOD = "size" # can be "order", "size" or "legacy"
FEATURES = "Norms" # Norms or Coords
PCA_COMPONENTS = None # set to None for Averaging method, set to integer to use PCA
NO_TREES = 100 # set to None for default (10)

# Set the mesh filename
meshFilename = "Asymknee13_boneSurface.vtk"
landmarksFilename = "Asymknee13.csv"
meshLocation = "bones"
landmarksFileLocation = meshLocation
mapLocation = "bones_markedLandmarksHeatmapTruthIndiv"
estimatorLocation = "estimators"

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[1]

# Check to see if it exists
subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
open(os.path.join(meshLocation, subBoneFilename), 'r')

# Load the mesh model and generate graph
model = read_mesh(os.path.join(meshLocation, subBoneFilename))
modelGraph, idArray, invIdArray = generate_graph(model)

# Load all the landmarks Point IDs for this mesh
landmarks = load_landmarks(os.path.join(landmarksFileLocation, landmarksFilename))
allLandmarkIds = get_landmark_ids(model, landmarks)
landmarkIds = []
for i, ID in enumerate(allLandmarkIds):
	if i in LANDMARK_REGIONS[subBone]:
		landmarkIds.append(ID)

# For each landmark, create an estimator to locate that landmark
for i, ID in enumerate(landmarkIds):
	actualLandmarkNo = LANDMARK_REGIONS[subBone][i]
	print("Creating estimator for landmark {} on file {}".format(actualLandmarkNo, subBoneFilename))
	if PATCH_METHOD == "order":
		patch_print = "Patch Order: {}".format(ORDER)
	elif PATCH_METHOD == "size":
		patch_print = "Patch Size: {}".format(PATCH_SIZE)
	print("Sample Rate: {}, {}".format(SAMPLE_RATE, patch_print))

	# Load the meshes with the distance information on them
	# This will be used to generate labels for the patches
	mapFilename = subBoneFilename.split(".")[0] + "_heatmap_landmark{}".format(actualLandmarkNo) + "." + subBoneFilename.split(".")[1]
	mapModel = read_mesh(os.path.join(mapLocation, mapFilename))
	mapScalars = mapModel.GetPointData().GetScalars()

	# Vectors to send to the estimator
	features = []
	labels = []

	# Sample points
	sampleIds = get_random_points(model, SAMPLE_RATE)
	progressPoints = [int(len(sampleIds)/10*(k+1)) for k in range(10)] # just to get some progress meter action

	# Go through each sample, create features and label for that sample
	for j, sID in enumerate(sampleIds):
		if j in progressPoints: # print out some progress
			print("About {} percent of {} sample points processed.".format((progressPoints.index(j)+1)*10, len(sampleIds)))
		if PATCH_METHOD == "order":
			samplePatch = create_patch(model, modelGraph, idArray, invIdArray, sID, ORDER)
		elif PATCH_METHOD == "size":
			samplePatch = create_patch_optimised(model, modelGraph, idArray, invIdArray, sID, PATCH_SIZE)

		sampleFeatures = get_features(model, samplePatch, FEATURES, PCA_COMPONENTS, sID)
		sampleLabel = mapScalars.GetValue(sID)
		if PCA_COMPONENTS != None:
			sampleFeatures = sampleFeatures.flatten()
		features.append(sampleFeatures)
		labels.append(sampleLabel)

		if j in progressPoints:
			print(os.path.join(mapLocation, mapFilename))
			print("Got feature {}, labelled {} for this".format(sampleFeatures, sampleLabel))
			if PCA_COMPONENTS != None:
				print(sampleFeatures.shape)

	features = np.array(features)	
	print(features.shape)

	# Create the estimator and fit it to the vectors
	if NO_TREES != None:
		estimator = RandomForestRegressor(n_estimators=NO_TREES)
	else:
		estimator = RandomForestRegressor()
	estimator.fit(features, labels)

	# Export the estimator
	meshBaseString = landmarksFilename.split(".")[0]
	landmarkString = "Landmark" + str(actualLandmarkNo)
	estimatorString = "RFR" + str(NO_TREES)
	if PATCH_METHOD == "order":
		orderString = "Order" + str(ORDER)
	elif PATCH_METHOD == "size":
		orderString = "Size" + str(PATCH_SIZE)
	sampleString = "SampleRate" + str(int(SAMPLE_RATE * 100))
	if PCA_COMPONENTS == None:
		featureString = "Features-Avg{}".format(FEATURES)
	else:
		featureString = "Features-All{}PCA{}".format(FEATURES, PCA_COMPONENTS)

	estimatorFilename = meshBaseString + "_" + landmarkString + "_" + estimatorString + "_" + orderString + "_" + sampleString + "_" + featureString + ".pkl"
	joblib.dump(estimator, os.path.join(estimatorLocation, estimatorFilename))