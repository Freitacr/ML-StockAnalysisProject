try:
	from math import log2
except ImportError:
	from math import log #assume using python2
	def log2(x):
		return log(x) / log(2)
	
class Node:
	
	#assumed data is given in form [[[X0], Y0] ... [[Xn], Yn]]
	
	def __init__(self, data, feature_index, used_feature_indices):
		self.path_mapping_dictionary = {}
		self.class_mapping_dictionary = {}
		self.data_list = []
		self.used_feature_indices = [feature_index]
		self.used_feature_indices.extend(used_feature_indices)
		self.next_nodes = []
		self.feature_index = feature_index
		if not len(data) == 0:	
			self.__map_data(data)
			self.__setup_lists()
			self.__add_data_to_lists(data)
			self.__setup_class_mapping(data)
			self.__entropy = self.__calcEntropy(len(data))
			self.__splitInfo = self.__calcSplitInfo(len(data))
		
	@classmethod
	def load (cls, file_handle):
		feature_index = int(file_handle.readline())
		dictionaryStringSplit = file_handle.readline().split(" ")
		num_next_nodes = int(file_handle.readline())
		ret = Node([], feature_index, [])
		for str in dictionaryStringSplit:
			entrySplit = str.split("=")
			ret.path_mapping_dictionary[entrySplit[0]] = int(entrySplit[1])
		for x in range(num_next_nodes):
			ret.next_nodes.extend([TempNode(ret, x)])
		return ret
		
	def __map_data(self, data):
		for x in data:
			currValue = x[0][self.feature_index]
			try:
				temp1 = float(currValue)
				temp2 = int(float(currValue))
				if not temp1 == temp2:
					raise TypeError("Feature " + str(currValue) + " is not discrete")
			except ValueError:
				pass
			if not currValue in self.path_mapping_dictionary:
				self.path_mapping_dictionary[currValue] = len(self.path_mapping_dictionary)
				
				
	def __setPathMappingDictionary(self, dictionary):
		self.path_mapping_dictionary = dictionary
				
	def __setup_lists(self):
		for x in range(len(self.path_mapping_dictionary)):
			self.data_list.extend([[]])
			self.next_nodes.extend([TempNode(self, x)])
			
	def __setup_class_mapping(self, data):
		for x in data:
			currValue = x[1]
			try:
				temp1 = float(currValue)
				temp2 = int(float(currValue))
				if not temp1 == temp2:
					raise TypeError("Label " + str(currValue) + " is not discrete")
			except ValueError:
				pass
			if not currValue in self.class_mapping_dictionary:
				self.class_mapping_dictionary[currValue] = len(self.class_mapping_dictionary)
	
	def __add_data_to_lists(self, data):
		for x in data:
			currTrainingExample = x[0]
			path_index = self.path_mapping_dictionary[currTrainingExample[self.feature_index]]
			self.data_list[path_index].extend([x])
			
	def getEntropy(self):
		return self.__entropy
	
	def getSplitInfo(self):
		return self.__splitInfo
	
	def __calcEntropyAtPath(self, path_id):
		amount_of_classes = [0] * len(self.class_mapping_dictionary)
		for set in self.getDataForPath(path_id):
			amount_of_classes[self.class_mapping_dictionary[set[1]]] = amount_of_classes[self.class_mapping_dictionary[set[1]]] + 1
		entropy = 0
		for x in amount_of_classes:
			if (len(self.getDataForPath(path_id)) == 0) or x == 0:
				continue
			else:
				entropy -= ( (x / len(self.getDataForPath(path_id))) * log2(x / len(self.getDataForPath(path_id))))
		return entropy
	
	def __calcEntropy(self, dataLen):
		entropy = 0
		for path in range(self.getNumPaths()):
			#print((len(self.getDataForPath(path)) / dataLen) * self.__calcEntropyAtPath(path))
			entropy += (len(self.getDataForPath(path)) / dataLen) * self.__calcEntropyAtPath(path)
		return entropy
		
	def __calcSplitInfo(self, dataLen):
		splitInfo = 0
		for path in range(self.getNumPaths()):
			splitInfo -= (len(self.getDataForPath(path)) / dataLen) * log2(len(self.getDataForPath(path)) / dataLen)
		return splitInfo
	
	def getNextNodes(self):
		return self.next_nodes
		
	def setNextNode(self, node, node_index):
		self.next_nodes[node_index] = node
	
	def getDataForPath(self, path_id):
		return self.data_list[path_id]
	
	def getUsedFeatures(self):
		return self.used_feature_indices
	
	def getNumPaths(self):
		return len(self.path_mapping_dictionary)
		
	def predict(self, X):
		value_to_index = X[self.feature_index]
		try:
			nextNode = self.getNextNodes()[self.path_mapping_dictionary[value_to_index]]
			return nextNode.predict(X)
		#if a feature is handed to a node and it hasn't seen it during training, then it will make its best effort to send it to a node that can handle it.
		#Best effort in this case means attempting to find a non-EndNode that has the capability to predict something in the example
		#Failing that, if an End Node is attatched to this node, it will use the EndNode with the lowest path index
		#If no End Nodes are attatched, then it simply sends it to the default (0 path index) node
		except KeyError:
			end_nodes = []
			for node in self.getNextNodes():
				if not type(node) == type(self):
					end_nodes.extend([node])
				else:
					node_feature_index = node.feature_index
					node_path_map = node.path_mapping_dictionary
					node_value = X[node_feature_index]
					if node_value in node_path_map:
						return node.predict(X)
			if len(end_nodes) > 0:
				return end_nodes[0].predict(X)
			else:
				return self.getNextNodes()[0].predict(X)
		
	def store(self, file_handle):
		file_handle.write("Node\n")
		file_handle.write(str(self.feature_index) + "\n")
		#store path dictionary
		dictString = ""
		for entry in self.path_mapping_dictionary.items():
			dictString += str(entry[0]) + "=" + str(entry[1]) + " "
		file_handle.write(dictString.rstrip() + "\n")
		file_handle.write(str(len(self.next_nodes)) + "\n")
		return True
		
class TempNode:
	
	def __init__(self, parent_node, path_index):
		self.parent = parent_node
		self.index = path_index
	
	def getParent(self):
		return self.parent
	def getIndex(self):
		return self.index
	
	
class EndNode:
	
	def __init__(self, data):
		if not len(data) == 0:
			self.data = data
			self.class_mapping_dictionary = {}
			self.__map_data(data)
			self.result_class = self.__calc_class(data)
		else:
			self.result_class = -1

	def __map_data(self, data):
		for x in data:
			currValue = x[1]
			if not currValue in self.class_mapping_dictionary:
				self.class_mapping_dictionary[currValue] = len(self.class_mapping_dictionary)
		
	def __calc_class(self, data):
		values = [0] * len(self.class_mapping_dictionary)
		for x in data:
			values[self.class_mapping_dictionary[x[1]]] = values[self.class_mapping_dictionary[x[1]]] + 1
		highest_class_amount = -1
		highest_class_index = -1
		for x in range(len(values)):
			if values[x] > highest_class_amount:
				highest_class_amount = values[x]
				highest_class_index = x
		for entry in self.class_mapping_dictionary.items():
			if entry[1] == highest_class_index:
				return entry[0]
		return None
			
	@classmethod
	def load (cls, file_handle):
		class_id = (file_handle.readline().rstrip())
		ret = EndNode([])
		ret.result_class = class_id
		return ret
		
	def getClass(self):
		return self.result_class
		
	def predict(self, X):
		return self.getClass()
		
	def store(self, file_handle):
		file_handle.write("End\n")
		file_handle.write(str(self.getClass()) + "\n")
	
class DecisionTree:
	
	def __init__(self):
		self.label_index_dictionary = {}
		self.__root = None
		
	def copy_settings(self):
		return DecisionTree()
		
	def load(self, fileName):
		self.__init__()
		model_file = None
		try:
			model_file = open(fileName, "r")
		except FileNotFoundError:
			return False
		#remove "Node" line. The first node must of class Node, otherwise it wouldn't be a "tree" at all, and also
		#wouldn't be able to make decisions as it had no EndNode
		model_file.readline()
		self.__root = Node.load(model_file)
		nodes_to_load = self.__root.getNextNodes()
		while not len(nodes_to_load) == 0:
			temp_nodes = []
			for node in nodes_to_load:
				nodeType = model_file.readline().rstrip()
				if nodeType == "Node":
					node.getParent().setNextNode(Node.load(model_file), node.getIndex())
				else:
					node.getParent().setNextNode(EndNode.load(model_file), node.getIndex())
				if type(node.getParent().getNextNodes()[node.getIndex()]) == type(self.__root):
					temp_nodes.extend(node.getParent().getNextNodes()[node.getIndex()].getNextNodes())
			nodes_to_load = temp_nodes
		
	def store(self, fileName):
		if self.__root == None:
			return False
		file = None
		try:
			file = open(fileName, "w")
		except FileNotFoundError:
			return False
		nodes_to_store = []
		self.__root.store(file)
		nodes_to_store.extend(self.__root.getNextNodes())
		while not len(nodes_to_store) == 0:
			temp_nodes = []
			for x in nodes_to_store:
				x.store(file)
				if type(x) == type(self.__root):
					temp_nodes.extend(x.getNextNodes())
			nodes_to_store = temp_nodes
		file.close()
		return False
		
	
	def train(self, X, Y):
		self.__init__()
		dat = []
		for index in range(len(X)):
			dat.extend([[X[index], Y[index]]])
		num_of_features = len(X[0])
		self.__map_labels(Y)
		root = None
		nodes_to_grow = []
		dataEntropy = self.__entropy(dat)
		trialNodes = []
		for x in range(num_of_features):
			trialNodes.extend([Node(dat, x, [])])
		self.__root = self.__selectBestNode(trialNodes, dataEntropy)
		nodes_to_grow = self.__root.getNextNodes()
		while not len(nodes_to_grow) == 0:
			temp_nodes = []
			for node in nodes_to_grow:
				currData = node.getParent().getDataForPath(node.getIndex())
				possible_indexes = [s not in node.getParent().getUsedFeatures() for s in range(num_of_features)]
				currTrialNodes = []
				for index in range(len(possible_indexes)):
					if possible_indexes[index]:
						currTrialNodes.extend([Node(currData, index, node.getParent().getUsedFeatures())])
				parent_entropy = node.getParent().getEntropy()
				node.getParent().setNextNode(self.__selectBestNode(currTrialNodes, parent_entropy), node.getIndex())
				if len(node.getParent().getUsedFeatures()) == num_of_features or node.getParent().getNextNodes()[node.getIndex()].getEntropy() <= .2:
					node.getParent().setNextNode(EndNode(node.getParent().getDataForPath(node.getIndex())), node.getIndex())
					curr_used = []
					curr_used.extend([node.getParent().getNextNodes()[node.getIndex()].getClass()])
					curr_used.extend(node.getParent().getUsedFeatures())
					continue
				temp_nodes.extend(node.getParent().getNextNodes()[node.getIndex()].getNextNodes())
			nodes_to_grow = temp_nodes
		
	def __selectBestNode(self, trial_nodes, parentEntropy):
		if len(trial_nodes) == 0:
			return None
		maxGainRatio = -1
		maxNodeIndex = -1
		for node_index in range(len(trial_nodes)):
			gainSplit = parentEntropy - trial_nodes[node_index].getEntropy()
			if trial_nodes[node_index].getSplitInfo() == 0:
				continue
			gainRatio = gainSplit / trial_nodes[node_index].getSplitInfo()
			if gainRatio > maxGainRatio:
				maxGainRatio = gainRatio
				maxNodeIndex = node_index
		return trial_nodes[maxNodeIndex]
		
	def predict(self, X):
		return self.__root.predict(X)
		
	def __entropy(self, dataset):
		amounts_of_classes = [0] * len(self.label_index_dictionary)
		for set in dataset:
			amounts_of_classes[self.label_index_dictionary[set[1]]] = amounts_of_classes[self.label_index_dictionary[set[1]]] + 1
		entropy = 0
		for x in amounts_of_classes:
			entropy -= ( (x / len(dataset)) * log2(x / len(dataset)))
		return entropy
		
	def __map_labels(self, Y):
		for label in Y:
			if not label in self.label_index_dictionary.keys():
				self.label_index_dictionary[label] = len(self.label_index_dictionary)