import numpy as np
from sklearn.decomposition import PCA

"""
Generate a feature array from a patch, where a patch is a list of point IDs.
"""
def get_features(model, patch, allNorms=False):
	pca_components = 3
	x_norms = []
	y_norms = []
	z_norms = []

	norms = model.GetPointData().GetNormals()

	for ID in patch:
		x_norms.append(norms.GetValue(ID*3))
		y_norms.append(norms.GetValue(ID*3+1))
		z_norms.append(norms.GetValue(ID*3+2))

	if allNorms == False:
		avg_x_norm = np.mean(x_norms)
		avg_y_norm = np.mean(y_norms)
		avg_z_norm = np.mean(z_norms)

		feat_set = []
		#feat_set.extend(x_norms)
		#feat_set.extend(y_norms)
		#feat_set.extend(z_norms)

		feat_set = [avg_x_norm, avg_y_norm, avg_z_norm]

	else:
		feat_set = []
		feat_set.append(x_norms)
		feat_set.append(y_norms)
		feat_set.append(z_norms)

		feat_set = np.array(feat_set)
		feat_set = feat_set.transpose() # rows are normals, columns are x,y,z

		feat_set -= np.mean(feat_set, axis=0)
		feat_set /= np.std(feat_set, axis=0)

		pca = PCA(n_components = pca_components)
		feat_set = pca.fit(feat_set).components_
		feat_set = feat_set.reshape(1, -1)

	return feat_set