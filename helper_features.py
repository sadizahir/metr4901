import numpy as np
from sklearn.decomposition import PCA

"""
Generate a feature array from a patch, where a patch is a list of point IDs.
"""
def get_features(model, patch, features, pca_components, ID):
	if features == "Norms":
		x_norms = []
		y_norms = []
		z_norms = []

		norms = model.GetPointData().GetNormals()

		for nID in patch:
			x_norms.append(norms.GetValue(nID*3))
			y_norms.append(norms.GetValue(nID*3+1))
			z_norms.append(norms.GetValue(nID*3+2))

	elif features == "Coords":
		x_norms = []
		y_norms = []
		z_norms = []

		centreX, centreY, centreZ = model.GetPoint(ID)

		for nID in patch:
			IDx, IDy, IDz = model.GetPoint(nID)
			x_norms.append(IDx - centreX)
			y_norms.append(IDy - centreY)
			z_norms.append(IDz - centreZ)

	if pca_components == None:
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