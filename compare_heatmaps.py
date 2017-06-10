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

ORDER = 5
SAMPLE_RATE = 0.2

# "Truth" heatmap details
mapLocation = "bones_markedLandmarksHeatmapTruthIndiv"
guessLocation = "bones_markedLandmarksHeatmapGuessIndiv"

meshFilename = "Asymknee22_boneSurface.vtk"
landmarksFilename = "Asymknee13.csv"

# Select sub-bone of interest
subBones = ["Patella", "Tibia", "Femur", ""]
subBone = subBones[1]

landmarkNos = LANDMARK_REGIONS[subBone]
for actualLandmarkNo in landmarkNos:
	subBoneFilename = meshFilename.split(".")[0] + subBone + "." + meshFilename.split(".")[1]
	mapFilename = subBoneFilename.split(".")[0] + "_heatmap_landmark{}".format(actualLandmarkNo) + "." + subBoneFilename.split(".")[1]
	open(os.path.join(mapLocation, mapFilename), 'r') # check to see if it exists

	guessFilename = mapFilename.split(".")[0] + "_guessedBy{}_Landmark{}_RFR_Order{}_SampleRate{}_Features-AvgNorms".format(landmarksFilename.split(".")[0], actualLandmarkNo, ORDER, int(SAMPLE_RATE*100)) + "." + mapFilename.split(".")[1]
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


