import numpy as np

"""
Generate a feature array from a patch, where a patch is a list of point IDs.
"""
def get_features(model, patch):
	x_norms = []
	y_norms = []
	z_norms = []

	norms = model.GetPointData().GetNormals()

	for ID in patch:
		x_norms.append(norms.GetValue(ID*3))
		y_norms.append(norms.GetValue(ID*3+1))
		z_norms.append(norms.GetValue(ID*3+2))

	avg_x_norm = np.mean(x_norms)
	avg_y_norm = np.mean(y_norms)
	avg_z_norm = np.mean(z_norms)

	# feat_set = []
	# feat_set.extend(x_norms)
	# feat_set.extend(y_norms)
	# feat_set.extend(z_norms)

	feat_set = [avg_x_norm, avg_y_norm, avg_z_norm]

	return feat_set