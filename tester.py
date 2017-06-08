"""
Uses a trained estimator to generate a heatmap. Or maybe a landmark.
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
from helper_patches import get_random_points
from helper_features import get_features
from constants import LANDMARK_REGIONS

SAMPLE_RATE = 0.2
ORDER = 5

# Set the mesh filename
meshFilename = "Asymknee22_boneSurface.vtk"
landmarksFilename = "Asymknee13.csv"
meshLocation = "bones"
landmarksFileLocation = meshLocation
mapLocation = "bones_markedLandmarksHeatmapTruthIndiv"
guessLocation = "bones_markedLandmarksHeatmapGuessIndiv"
estimatorLocation = "estimators"

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[1]

writer = vtk.vtkPolyDataWriter()

landmarkNos = LANDMARK_REGIONS[subBone]
for actualLandmarkNo in landmarkNos:
	# Load the estimator
	meshBaseString = landmarksFilename.split(".")[0]
	landmarkString = "Landmark" + str(actualLandmarkNo)
	estimatorString = "RFR"
	orderString = "Order" + str(ORDER)
	sampleString = "SampleRate" + str(int(SAMPLE_RATE * 100))
	featureString = "Features-AvgNorms"

	estimatorFilename = meshBaseString + "_" + landmarkString + "_" + estimatorString + "_" + orderString + "_" + sampleString + "_" + featureString + ".pkl"
	estimator = joblib.load(os.path.join(estimatorLocation, estimatorFilename))

	subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
	open(os.path.join(meshLocation, subBoneFilename), 'r') # check to see if mesh exists

	print("Using estimator {} on file {}".format(estimatorFilename, subBoneFilename))

	# Load the mesh model and generate graph
	model = read_mesh(os.path.join(meshLocation, subBoneFilename))
	modelGraph, idArray, invIdArray = generate_graph(model)

	# Create a scalar array which will contain the estimated heatmap
	scalars = vtk.vtkDoubleArray()
	scalars.SetNumberOfValues(model.GetNumberOfPoints())

	# Create a temporary array containing competing estimates for each point
	scalarCandidates = [[] for i in range(model.GetNumberOfPoints())]
	scalarAverageCandidates = [-1 for i in range(model.GetNumberOfPoints())]

	# Sample points
	sampleIds = get_random_points(model, SAMPLE_RATE)
	progressPoints = [int(len(sampleIds)/10*(k+1)) for k in range(10)] # just to get some progress meter action

	# Go through each sample, create features and label for that sample
	for j, sID in enumerate(sampleIds):
		if j in progressPoints: # print out some progress
			print("About {} percent of {} sample points processed.".format((progressPoints.index(j)+1)*10, len(sampleIds)))
		samplePatch = create_patch(model, modelGraph, idArray, invIdArray, sID, ORDER)
		sampleFeatures = get_features(model, samplePatch)

		# generate a label using the estimator
		sampleLabel = estimator.predict(np.array(sampleFeatures).reshape(1, -1))

		# apply to scalars
		for pID in samplePatch:
			scalarCandidates[pID].append(sampleLabel)

		if j in progressPoints:
			print("Got feature {}, guessed around {} for this".format(sampleFeatures, sampleLabel))

	# Smooth out the point information from the candidate estimates
	for i, candidate in enumerate(scalarCandidates):
		if candidate != []:
			scalarAverageCandidates[i] = np.median(np.array(candidate))

	# Get rid of outliers if they exist, and apply to actual scalar field
	for i, averageCandidate in enumerate(scalarAverageCandidates):
		if averageCandidate == -1:
			averageCandidate = max(scalarAverageCandidates)
		scalars.SetValue(i, averageCandidate)

	# Output the estimated heatmap
	model.GetPointData().SetScalars(scalars)
	outputFilename = subBoneFilename.split(".")[0] + "_heatmap_landmark{}_guessedBy{}".format(actualLandmarkNo, estimatorFilename.split(".")[0]) + "." + subBoneFilename.split(".")[1]
	writer.SetFileName(os.path.join(guessLocation, outputFilename))
	writer.SetInputData(model)
	writer.Write()

