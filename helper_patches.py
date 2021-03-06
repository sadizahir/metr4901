import vtk
import random

"""
Given a model, associated graph, and a point ID, generate a list of IDs
which represent the "neighbourhood" around and including that point.

The "size" of the neighbourhood is given by how deeply we compute "neighbours
of neighbours" (the "order"). For example, zeroth-order is the point itself.
First-order is the point's neighbours, second-order is the neighbours'
neighbours, and so on.

"""
def create_patch(model, modelGraph, idArray, invIdArray, ID, order):
	neighbourhood = []
	neighbourhood.append(ID)

	if order == 0: # we just return the point of interest
		return neighbourhood

	iterator = vtk.vtkAdjacentVertexIterator()
	modelGraph.GetAdjacentVertices(idArray.GetValue(ID), iterator)

	while iterator.HasNext():
		nextVertex = iterator.Next()
		nextPointID = invIdArray.GetValue(nextVertex)
		neighbourhood.append(nextPointID)
		furtherPoints = create_patch(model, modelGraph, idArray, invIdArray, nextPointID, order-1)
		neighbourhood.extend(furtherPoints)

	neighbourhood = list(set(neighbourhood))

	return neighbourhood

def create_patch_optimised(model, modelGraph, idArray, invIdArray, ID, size):
	neighbourhood = []

	candidates = vtk.vtkDoubleArray()
	candidates.SetNumberOfValues(model.GetNumberOfPoints())
	dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
	dijkstra.SetInputData(model)
	dijkstra.SetStartVertex(ID)
	dijkstra.SetEndVertex(0) # just a random vertex so that it will compute all the paths
	dijkstra.Update()
	dijkstra.GetCumulativeWeights(candidates)

	for i in range(model.GetNumberOfPoints()):
		weight = candidates.GetValue(i)
		if weight <= size:
			neighbourhood.append(i)

	return neighbourhood

def get_random_points(model, sampleRate):
	noOfPoints = model.GetNumberOfPoints()
	noOfSamples = int(noOfPoints * sampleRate)
	sampleIDs = random.sample(range(int(noOfPoints*0.99)), noOfSamples) # for some reason, VTK will crash if you try to sample the last dozen or so points

	return sampleIDs