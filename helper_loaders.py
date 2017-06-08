import vtk

"""
Reads a PolyData and produces a model with norms/links.
"""
def read_mesh(filename):
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(filename)
	reader.Update()

	model = reader.GetOutput()
	model.BuildLinks()

	norms = vtk.vtkPolyDataNormals()
	norms.SetInputData(model)
	norms.Update()
	norms.FlipNormalsOn()
	norms.Update()
	model = norms.GetOutput()
	model.BuildLinks()

	return model

"""
Generates a graph of a model, as well as PointID arrays.
"""
def generate_graph(model):
	# extract edges
	extractEdges = vtk.vtkExtractEdges()
	extractEdges.SetInputData(model)
	extractEdges.Update()

	# alias edge and point data for the edges
	edgeLines = extractEdges.GetOutput().GetLines()
	edgePoints = extractEdges.GetOutput().GetPoints()
	NoOfPoints = edgePoints.GetNumberOfPoints()

	# construct edge graph
	modelGraph = vtk.vtkMutableUndirectedGraph()

	# create vertices of graph
	# firstly, keep lookup table of Point Ids
	idArray = vtk.vtkIdTypeArray() # lookup table of Ids
	idArray.SetName("Id_Lookup")
	idArray.SetNumberOfTuples(NoOfPoints)

	invIdArray = vtk.vtkIdTypeArray() # inverse lookup table of Ids
	invIdArray.SetName("Inverse_Id_Lookup")
	invIdArray.SetNumberOfTuples(NoOfPoints)

	for j in range(NoOfPoints):
	    v = modelGraph.AddVertex()
	    vDash = extractEdges.GetLocator().IsInsertedPoint(model.GetPoint(v))
	    idArray.SetValue(v, vDash)
	    invIdArray.SetValue(vDash, v)
	    
	# traverse all of the edges and construct undirected graph of mesh vertices
	for j in range(edgeLines.GetNumberOfCells()):
	    edge = vtk.vtkLine.SafeDownCast(extractEdges.GetOutput().GetCell(j))
	    modelGraph.AddGraphEdge(edge.GetPointIds().GetId(0), edge.GetPointIds().GetId(1))
	    
	modelGraph.SetPoints(edgePoints)

	return modelGraph, idArray, invIdArray

"""
Load landmark co-ordinates from a file and store them as tuples in a list.
"""
def load_landmarks(filename):
	landmarksFp = open(filename, 'r')
	landmarks = []
	for lineNo, line in enumerate(landmarksFp.readlines()):
		if lineNo != 0: # skip first line
			landmarks.append(tuple([float(i) for i in line.split(",")]))
	return landmarks

"""
Generates list of landmark Point IDs from landmark co-ordinates.
"""
def get_landmark_ids(model, landmarks):
	landmarkIds = []
	for coord in landmarks:
		coord = coord[1:] # chop off the landmark no.
		landmarkIds.append(model.FindPoint(*coord))
	return landmarkIds