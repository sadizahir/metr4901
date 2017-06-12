"""
Compares points by outputting vectors for each!
"""
from __future__ import print_function

import os
import csv
import math
import vtk

from helper_loaders import read_mesh
from helper_loaders import generate_graph
from helper_loaders import load_landmarks
from helper_loaders import get_landmark_ids
from helper_patches import create_patch
from constants import LANDMARK_REGIONS

SAMPLE_RATE = 0.2
ORDER = 7
PATCH_SIZE = 14
PATCH_METHOD = "size" # can be "order", "size" or "legacy"
FEATURES = "Norms" # Norms or Coords
PCA_COMPONENTS = None # set to None for Averaging method, set to integer to use PCA
NO_TREES = 10 # set to None for default (10)

# "Truth" heatmap details
mapLocation = "bones_markedLandmarksPointsTruthIndiv"
guessLocation = "bones_markedLandmarksPointsGuessIndiv"

basePrefix = "Asymknee"
meshFilename = "Asymknee22_boneSurface.vtk"
landmarksFilenames = ["Asymknee11.csv", "Asymknee13.csv"]

estimatorString = "RFR" + str(NO_TREES)

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[1]

pairwisePoints = [("Landmark No.", "Euclidean Distance", "Geodesic Distance")]

landmarkNos = LANDMARK_REGIONS[subBone]
for actualLandmarkNo in landmarkNos:
	meshBaseString = ""
	for landmarksFilename in landmarksFilenames:
		meshBaseString += landmarksFilename.split(".")[0][len(basePrefix):] + ","
	meshBaseString = meshBaseString[:-1]

	subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
	mapFilename = subBoneFilename.split(".")[0] + "_point_landmark{}".format(actualLandmarkNo) + "." + subBoneFilename.split(".")[1]
	open(os.path.join(mapLocation, mapFilename), 'r') # check to see if it exists

	if PCA_COMPONENTS == None:
		featureString = "Features-Avg{}".format(FEATURES)
	else:
		featureString = "Features-All{}PCA{}".format(FEATURES, PCA_COMPONENTS)

	if PATCH_METHOD == "order":
		orderString = "Order{}".format(ORDER)
	elif PATCH_METHOD == "size":
		orderString = "Size{}".format(PATCH_SIZE)

	guessFilename = mapFilename.split(".")[0] + "_guessedBy{}_Landmark{}_{}_{}_SampleRate{}_{}".format(meshBaseString, actualLandmarkNo, estimatorString, orderString, int(SAMPLE_RATE*100), featureString) + "." + mapFilename.split(".")[1]
	open(os.path.join(guessLocation, guessFilename), 'r') # check to see if it exists

	mapModel = read_mesh(os.path.join(mapLocation, mapFilename))
	guessModel = read_mesh(os.path.join(guessLocation, guessFilename))

	mapScalars = mapModel.GetPointData().GetScalars()
	guessScalars = guessModel.GetPointData().GetScalars()

	mapPoint = -1
	for i in range(mapModel.GetNumberOfPoints()):
		if mapScalars.GetValue(i) == 2:
			mapPoint = i
			break

	guessPoint = -1
	for i in range(guessModel.GetNumberOfPoints()):
		if guessScalars.GetValue(i) == 2:
			guessPoint = i
			break

	mapCoords = mapModel.GetPoint(mapPoint)
	guessCoords = mapModel.GetPoint(guessPoint)

	euclideanDistance = math.sqrt((mapCoords[0] - guessCoords[0])**2 + (mapCoords[1] - guessCoords[1])**2 + (mapCoords[2] - guessCoords[2])**2)

	dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
	dijkstra.SetInputData(mapModel)
	dijkstra.SetStartVertex(mapPoint)
	dijkstra.SetEndVertex(guessPoint)
	dijkstra.Update()
	weights = vtk.vtkDoubleArray()
	dijkstra.GetCumulativeWeights(weights)

	geodesicDistance = weights.GetValue(guessPoint)

	pairwisePoints.append((actualLandmarkNo, euclideanDistance, geodesicDistance))



csvFilename = guessFilename.split(".")[0] + "_comparedAllPoints.csv"

with open(csvFilename, 'wb') as csvfile:
	writer = csv.writer(csvfile)

	for ps in pairwisePoints:
		writer.writerow(ps)


