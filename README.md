# metr4901
For this software to work, you need to create directories which contain the bones you are interested, pre-processed into separated meshes (e.g. femur, tibia, patella).

You need directories which look like this, but can be renamed inside the script files if necessary:

bones --> this is where the basic boneSurface meshes are, as well as the landmark CSVs
bones_markedLandmarksHeatmapGuessIndiv --> this is where the estimated heatmaps will be saved
bones_markedLandmarksHeatmapTruthIndiv --> this is where the ground truth heatmaps will be saved
bones_markedLandmarksPointsGuessIndiv --> this is where the estimated landmark points will be saved
bones_markedLandmarksPointsTruthIndiv --> this is where the ground truth landmark points will be saved
estimators --> this is where the estimators will be saved

In order:
Run landmark_locator.py to record which landmarks appear in which subregions, if not testing on knees. Record these positions in the LANDMARK_REGIONS constant inside constants.py.

Run landmark_patches_indiv.py to create the ground truth landmark points.

Run landmark_pathmaker.py to create the ground truth heatmaps.

Run trainer_multi.py to create an estimator with the parameters set at the beginning of the script.

Run tester_multi.py to use an estimator on a mesh to produce heat map and point predictions.

Run compare_points_multi.py to generate a comparison report on the landmark point predictions.

If also interested in heatmap statistics, you can also run compare_heatmaps_multi.py.
