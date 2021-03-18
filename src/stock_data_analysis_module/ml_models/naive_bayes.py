import os
import sys
import operator




class NaiveBayes:

	def __init__(self, assume_all_features_known = True):
		self.class_mapping_dictionary = {}
		self.feature_mapping_dictionaries = []
		self.feature_probability_tables=[]
		self.class_probability_table=[]
		self.assume_all_features_known = assume_all_features_known

	def copy(self):
		ret_model = NaiveBayes(self.assume_all_features_known)
		ret_model.class_mapping_dictionary = self.class_mapping_dictionary.copy()
		for self_dict in self.feature_mapping_dictionaries:
			ret_model.feature_mapping_dictionaries.append(self_dict.copy())
		for self_dict in self.feature_probability_tables:
			ret_model.feature_probability_tables.append(self_dict.copy())
		ret_model.class_probability_table = self.class_probability_table.copy()
		return ret_model
		
	def copy_settings(self):
		return NaiveBayes(self.assume_all_features_known)
		
	def train(self, X, Y):
		#Generates the feature and label mapping dictionaries
		self.__setup_Mapping(X,Y)
		#Generates placeholder tables for the values to live in
		self.__setup_probability_tables()
		#Iterate over all training examples
		num_training_examples = len(X)
		for curr_example_index in range(num_training_examples):
			#Retrieve the mapping of Y[curr_example_index] from the class dictionary.
			label_index = self.__index_from_label(str(Y[curr_example_index]), self.class_mapping_dictionary)
			#Increment the numerator of the cell in class_probability_table that is referenced by the label index 
			self.__incr_row(self.class_probability_table, label_index)
			#Now comes time for the feature array for this example to be processed
			for feature_index in range(len(X[curr_example_index])):
				#Setup reference to the dictionary for the current variable
				curr_dictionary = self.feature_mapping_dictionaries[feature_index]
				#Setup reference to the feature table for the current variable 
				current_feature_table = self.feature_probability_tables[feature_index]
				#Retrieves the current value of the variable in the current training example
				current_feature_value = str(X[curr_example_index][feature_index])
				#get the index from the dictionary that belongs to the current feature value, then grab the row that specifies from the current_feature_table. Then increment the numerator in the cell that is referenced by label_index. 
				#or add one to the numerator of current_feature_table[index_from_feature][label_index]
				self.__incr_row(
				current_feature_table[self.__index_from_label(
				current_feature_value, curr_dictionary
				)], 
				label_index
				)
		#Set all the denominators in all of the feature_probability_tables
		for table in self.feature_probability_tables:
			self.__set_table_bottom(table)
		#Set the denominators in the class_probability_table
		self.__set_row_bottom(self.class_probability_table, len(Y))
			
			
	#Stores the model into a file specified by filename
	def store(self, filename):
		file = None
		try:
			file = open(filename, "w")
		except FileNotFoundError:
			return False
		file.write(str(len(self.feature_mapping_dictionaries)) + "\n")
		temp_string = ""
		for x in self.class_mapping_dictionary.items():
			temp_string += (str(x[0]) + "=" + str(x[1]) + " ")
		file.write(temp_string.rstrip())
		file.write("\n")
		for dict in self.feature_mapping_dictionaries:
			temp_string = ""
			for x in dict.items():
				temp_string += (str(x[0]) + "=" + str(x[1]) + " ")
			file.write(temp_string.rstrip())
			file.write("\n")
		temp_string = ""
		for x in self.class_probability_table:
			temp_string += str(x) + " "
		file.write(temp_string.rstrip())
		file.write("\n")
		for table in self.feature_probability_tables:
			for row in table:
				temp_string = ""
				for cell in row:
					temp_string += cell + " "
				file.write(temp_string.rstrip() + "\n")
		file.close()
		return True

	#Loads the model from a file specified by filename. Will only work if the file was created by store
	def load(self, filename):
		self.__init__()
		try:
			file = open(filename, "r")
			num_of_feature_dicts = int(file.readline())
			dict_split = file.readline().split(" ")
			for x in range(len(dict_split)):
				entry_split = dict_split[x].split("=")
				self.class_mapping_dictionary[entry_split[0]] = int(entry_split[1].rstrip())
			for dict_index in range((num_of_feature_dicts)):
				temp_dict = {}
				dict_split = file.readline().split(" ")
				for x in range(len(dict_split)):
					entry_split = dict_split[x].split("=")
					temp_dict[entry_split[0]] = int(entry_split[1].rstrip())
				self.feature_mapping_dictionaries.append(temp_dict)
			table_split = file.readline().split(" ")
			self.__setup_probability_tables()
			for x in range(len(table_split)):
				self.class_probability_table[x] = table_split[x].rstrip()
			for feature_dict_index in range(num_of_feature_dicts):
				num_cols = len(self.class_mapping_dictionary)
				num_rows = len(self.feature_mapping_dictionaries[feature_dict_index])
				for row in range(num_rows):
					table_split = file.readline().split(" ")
					for col in range(num_cols):
						self.feature_probability_tables[feature_dict_index][row][col] = table_split[col].rstrip()
		except (IndexError, FileNotFoundError):
			return False

	#Checks if the label exists in the dictionary passed, and returns its value. If not, returns -1
	def __index_from_label(self, label, dictionary):
		tuples = dictionary.items()
		for tuple in tuples:
			if tuple[0] == label:
				return tuple[1]
		return -1

	#Increments the numerator of the cell(referenced by index_to_inc) in the row passed
	def __incr_row(self, row, index_to_inc):
		#since cells are stored as "numerator/denominator", split them
		split = row[index_to_inc].split("/")
		#increase numerator
		split[0] = str(int(split[0]) + 1)
		#reassemble the cell's contents
		row[index_to_inc] = split[0] + "/" + split[1]

	#Sets the row's denominator to the value passed. 
	def __set_row_bottom(self, row, value):
		for cell_index in range(len(row)):
			cell_split = row[cell_index].split("/")
			cell_split[1] = str(value)
			row[cell_index] = cell_split[0] + "/" + cell_split[1]
		
	#Since all columns in the table should have the same denominator, and this will be the same as the numerator of 
	#the entry in class_probability_table with the same column index, this updates the table's denominators to reflect that.
	def __set_table_bottom(self, table):
		for row in table:
			for cell_index in range(len(row)):
				cell_split = row[cell_index].split("/")
				cell_split[1] = self.class_probability_table[cell_index].split("/")[0]
				row[cell_index] = cell_split[0] + "/" + cell_split[1]
	
	#Each time this method encounters a new label, it updates the corresponding dictionary, assigning it the next number in the list,
	#starting at 0
	def __setup_Mapping(self, X,Y):
		for label in Y:
			#error checking to ensure that this program does not attempt to use continuous values as features or classes
			try:
				int(label)
			except ValueError:
				try:
					float(label)
				except ValueError:
					pass
				else:
					raise TypeError("Label " + label + " is not a discrete label")
			if len(self.class_mapping_dictionary) == 0:
				self.class_mapping_dictionary[str(label)] = 0
			else:
				if label in self.class_mapping_dictionary.keys():
					continue
				else:
					self.class_mapping_dictionary[str(label)] = len(self.class_mapping_dictionary)

					
		#Add amount_of_features of tables into the mapping tables. Allowing space to save similar tables to
		#the one done above in one list.
		for feature_index in range(len(X[0])):
			self.feature_mapping_dictionaries.append({})
		
		for feature_section in X:
			for feature_index in range(len(feature_section)):
				#if the table for the current feature is empty, then push the first value into it. 
				try:
					int(feature_section[feature_index])
				except ValueError:
					try:
						float(feature_section[feature_index])
					except ValueError:
						pass
					else:
						raise TypeError("Feature " + str(feature_section[feature_index]) + " is not discrete")
				if len(self.feature_mapping_dictionaries[feature_index]) == 0:
					self.feature_mapping_dictionaries[feature_index][str(feature_section[feature_index])] = 0
				else:
					if feature_section[feature_index] in self.feature_mapping_dictionaries[feature_index].keys():
						continue
					else:
						self.feature_mapping_dictionaries[feature_index][str(feature_section[feature_index])] = len(self.feature_mapping_dictionaries[feature_index])
	#Actually performs the prediction following the naive bayes algorithm.
	def predict(self, x):
		prediction_probabilites = self.__sub_predict(x)
		highest_value = 0
		index = 0
		for x in range(len(prediction_probabilites)):
			if prediction_probabilites[x] > highest_value:
				index = x
				highest_value = prediction_probabilites[x]
		for entry in self.class_mapping_dictionary.items():
			if entry[1] == index:
				return entry[0]
		return None
		
	#not used in current testing implementation, however this does work. 
	def predict_top_n(self, x, n):
		prediction_probabilites = self.__sub_predict(x)
		prediction_probabilites_dictionary = {}
		for x in range(len(prediction_probabilites)):
			prediction_probabilites_dictionary[x] = prediction_probabilites[x]
		indices_sorted = sorted(prediction_probabilites_dictionary.items(), key=operator.itemgetter(1))
		ret = []
		if n > len(indices_sorted):
			n = len(indices_sorted)
		
		for x in range(n):
			for entry in self.class_mapping_dictionary.items():
				if entry[1] == indices_sorted[-x-1][0]:
					ret.append(entry[0])
		return ret

	#prediction method that obtains the conditional probabilities, and then multiplies them.
	#also accounts for the case of a particular conditional probability being zero, and handles the situation according to the Naive Bayes algorithm
	def __sub_predict(self, x):
		feature_indices = []
		for feature_index in range(len(x)):
			feature_indices.append(self.__index_from_label(str(x[feature_index]), self.feature_mapping_dictionaries[feature_index]))
		if -1 in feature_indices and self.assume_all_features_known:
			raise TypeError("One of the values provided in " + str(x) + " has never been seen as a feature before. This is different than the feature being known, but one state having never been seen before.")
		prediction_probabilites = []
		for label_index in range(len(self.class_mapping_dictionary)):
			curr_probability = 1.0
			for feature_index in range(len(feature_indices)):
				if not feature_indices[feature_index] == -1:
					split = self.feature_probability_tables[feature_index][feature_indices[feature_index]][label_index].split("/")
					division_res = float(split[0]) / float(split[1])
					#This is where the case is handled if a conditional probability is zero
					if (division_res == 0):
						division_res = self.__shadow_update(self.feature_probability_tables[feature_index], label_index, feature_indices[feature_index]);
					curr_probability *= division_res
				elif feature_indices[feature_index] == -1:
					curr_probability *= self.__unknown_update(self.feature_probability_tables[feature_index], label_index)
			prediction_probabilites.append(curr_probability)
		return prediction_probabilites
		
	
	#Generates the placeholder tables for class_probability_table & feature_probability_tables
	def __setup_probability_tables(self):
		self.class_probability_table = ["0/0"] * len(self.class_mapping_dictionary)
		num_feature_dictionaries = len(self.feature_mapping_dictionaries)
		for curr_dictionary_index in range(num_feature_dictionaries):
			temp_feature_probability_table = []
			for num in range(len(self.feature_mapping_dictionaries[curr_dictionary_index])):
				temp_feature_probability_table.append(["0/0"] * len(self.class_mapping_dictionary))
			self.feature_probability_tables.append(temp_feature_probability_table)

	#In the event that a conditional probability equals zero when calculations need to occur, this method will simulate if one more example were given that had all the possible values the variable could have taken, and return that value.
	#AKA (0 + (1 / number of possible values)) / current_denominator + 1
	def __shadow_update(self, table, col_num, row_num):
		to_add = 1 / len(table)
		split = table[row_num][col_num].split("/")
		return (float(split[0]) + to_add) / (float(split[1]) + 1)
		
		
	#This is for the case in which a feature that has NEVER been encountered before (like if we are representing
	# 10 different types models weather outside {i.e. rainy, overcast, sunny, etc} as the numbers 1-10
	# and the number 11 comes up when it comes time to predict. Naturally we can almost count this as just
	# a weird new type of virtual training example, controlled by a boolean value as to whether it should be used.
	def __unknown_update(self, table, col_num):
		to_add = 1 / (len(table) + 1)
		split = table[0][col_num].split("/")
		return (to_add) / (float(split[1]) + 1)
		

