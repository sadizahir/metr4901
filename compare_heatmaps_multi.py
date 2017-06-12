"""
Compares heatmaps by outputting vectors for each!
"""
from __future__ import print_function

import os
import csv

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
mapLocation = "bones_markedLandmarksHeatmapTruthIndiv"
guessLocation = "bones_markedLandmarksHeatmapGuessIndiv"

basePrefix = "Asymknee"
meshFilename = "Asymknee22_boneSurface.vtk"
landmarksFilenames = ["Asymknee11.csv", "Asymknee13.csv"]

estimatorString = "RFR" + str(NO_TREES)

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[2]

landmarkNos = LANDMARK_REGIONS[subBone]
for actualLandmarkNo in landmarkNos:
	meshBaseString = ""
	for landmarksFilename in landmarksFilenames:
		meshBaseString += landmarksFilename.split(".")[0][len(basePrefix):] + ","
	meshBaseString = meshBaseString[:-1]

	subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
	mapFilename = subBoneFilename.split(".")[0] + "_heatmap_landmark{}".format(actualLandmarkNo) + "." + subBoneFilename.split(".")[1]
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
	pairwise_scalars = []

	for i in range(mapModel.GetNumberOfPoints()):
		pairwise_scalars.append((i, mapScalars.GetValue(i), guessScalars.GetValue(i)))

	csvFilename = guessFilename.split(".")[0] + "_compared.csv"

	with open(csvFilename, 'wb') as csvfile:
		writer = csv.writer(csvfile)

		for ps in pairwise_scalars:
			writer.writerow(ps)


