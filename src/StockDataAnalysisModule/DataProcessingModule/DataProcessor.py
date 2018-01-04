'''
Created on Dec 24, 2017

@author: Colton Freitas
'''

from .DataRetrievalModule.DataRetriever import DataRetriever
from StockDataAnalysisModule.DataProcessingModule import DataProcessingUtils
from GeneralUtils.EPrint import eprint
from asyncio.futures import InvalidStateError

class DataProcessor:
    
    def __init__(self, login_credentials):
        self.login_credentials = login_credentials
    
    def __gen_training_examples(self, max_number_of_days_per_example, input_data, output_data):
        '''Generates training examples for the models in MLModels to use 
           @param max_number_of_days_per_example: The maximum number of days of input data to use in one example
           @param input_data: The list of data to be used as the input, or X portion of the return
           @param output_data: The list of data to be used as the classification or Y portion of the return 
        '''
        #This is a check to ensure that the output data will match correctly with the number of examples created from the input data
        if not (len(input_data[0][1])) - max_number_of_days_per_example == len(output_data[0][1]):
            raise InvalidStateError("Input size mismatch: expected %s, but got %s" % ((len(input_data[0][1])) - max_number_of_days_per_example, len(output_data[0][1])))
        #assumed input data in format [ [ticker, in_data] ... ]
        #assumed output data in format [ [ticker, out_data] ... ]
        ret_dict = {}
        for ticker_index in range(len(input_data)):
            X, Y = [[], []]
            in_data = input_data[ticker_index][1]
            class_data = output_data[ticker_index][1]
            #len(in_data) - max_number_of_days_per_example is the number of examples generated
            for example_index in range(len(in_data) - max_number_of_days_per_example):
                #If values in in_data are not lists
                if not type(in_data[example_index]) == type([]):
                    X.extend(in_data[example_index:example_index + max_number_of_days_per_example])
                #else combine all of the sublists for the days into one larger, one dimensional, list
                else:
                    full_list = []
                    for sublist in in_data[example_index:example_index + max_number_of_days_per_example]:
                        full_list.extend(sublist)
                    X.extend([full_list])
                Y.extend([class_data[example_index]])
            ret_dict[input_data[ticker_index][0]] = [X, Y]
        return ProcessedDataHolder(ret_dict)   
    
    def getPercentageChanges(self, column_list, max_number_of_days_per_example = 5):
        '''Formats the available data by calculating the percentage change between each day
           @param column_list: A list of the columns to obtain data from while obtaining data
           @param max_number_of_days_per_example: If this parameter is greater than zero, then it is the maximum number
                of days for each example when training examples are generated
                When it is below or equal to zero, then it is a flag to return the formatted data without transformation
                Into the training data
        '''
        data_retriever = DataRetriever(self.login_credentials, column_list)
        retrieved_data = data_retriever.getData()
        
        
        for ticker, data in retrieved_data:
            index_arr = list(range(-(len(data) - 1), 0))
            index_arr.reverse()
            for index in index_arr:
                #if volume == True data[index] should be (adj_close, volume_data)
                data[index] = list(data[index])
                for data_type_index in range(len(data[index])):
                    day1 = data[index - 1][data_type_index]
                    day2 = data[index][data_type_index]
                    #convert the data from a string to float, if that fails, then this whole thing will fail
                    try:
                        if type(day1) == type(""):
                            day1 = float(day1)
                        if type(day2) == type(""):
                            day2 = float(day2)
                    except ValueError:
                        eprint("String value found in data, ignoring. Prepare for new exception.")
                    data[index][data_type_index] = DataProcessingUtils.percentChangeBetweenDays(day1, day2)
            data[0] = list(data[0])
            for data_type_index in range(len(data[0])):
                #This is valid because there is nothing to compare the first day in the stock's history against but itself, so the change HAS to be zero.
                data[0][data_type_index] = 0
            
        #max_number_of_days_per_example flag checking
        if max_number_of_days_per_example <= 0:
            return retrieved_data
        else:
            input_data, target_data =[[],[]]
            for data in retrieved_data:
                #in_data = []
                tar_data = []
                ticker = data[0]
                data_arr = data[1]
                for x in range(len(data_arr) - max_number_of_days_per_example):
                    tar_data.extend([data_arr[x+max_number_of_days_per_example][0]])
                input_data.extend([[ticker, data_arr]])
                target_data.extend([[ticker, tar_data]])
            
            return self.__gen_training_examples(max_number_of_days_per_example, input_data, target_data)
        
    
     
    def getMovementDirections(self, column_list, max_number_of_days_per_example = 5):
        '''Formats the available data by using the percentage change between days and simplifying the result
           @param column_list: A list of the columns to obtain data from while obtaining data
           @param max_number_of_days_per_example: If this parameter is greater than zero, then it is the maximum number
                of days for each example when training examples are generated
                When it is below or equal to zero, then it is a flag to return the formatted data without transformation
                Into the training data
        '''
        retrieved_data = self.getPercentageChanges(column_list, max_number_of_days_per_example=-1)
        for x in retrieved_data:
            for data_point in x[1]:
                for data_type_index in range(len(data_point)):
                    if data_point[data_type_index] >= 1:
                        data_point[data_type_index] = 'up'
                    elif data_point[data_type_index] <= -1:
                        data_point[data_type_index] = 'down'
                    else:
                        data_point[data_type_index] = 'stag'
        
        if max_number_of_days_per_example <= 0:
            return retrieved_data
        else:
            input_data, target_data =[[],[]]
            for data in retrieved_data:
                #in_data = []
                tar_data = []
                ticker = data[0]
                data_arr = data[1]
                for x in range(len(data_arr) - max_number_of_days_per_example):
                    tar_data.extend([data_arr[x+max_number_of_days_per_example][0]])
                input_data.extend([[ticker, data_arr]])
                target_data.extend([[ticker, tar_data]])
            
            return self.__gen_training_examples(max_number_of_days_per_example, input_data, target_data)

    def getLimitedNumericalChange(self, column_list, max_number_of_days_per_example = 5):
        '''Formats the available data by calculating the percentage change between each day and integer casting the results
           @param column_list: A list of the columns to obtain data from while obtaining data
           @param max_number_of_days_per_example: If this parameter is greater than zero, then it is the maximum number
                of days for each example when training examples are generated
                When it is below or equal to zero, then it is a flag to return the formatted data without transformation
                Into the training data
        '''
        retrieved_data = self.getPercentageChanges(column_list, max_number_of_days_per_example=-1)
        
        for x in retrieved_data:
            for data_point in x[1]:
                for data_type_index in range(len(data_point)):
                    
                    if int(data_point[data_type_index]) >= 6:
                        data_point[data_type_index] = '5'
                    elif int(data_point[data_type_index]) <= -6:
                        data_point[data_type_index] = '-5'
                    else:
                        data_point[data_type_index] = str(int(data_point[data_type_index]))
        
        if max_number_of_days_per_example <= 0:
            return retrieved_data
        else:
            input_data, target_data = [[], []]
            for data in retrieved_data:
                tar_data = []
                ticker = data[0]
                data_arr = data[1]
                for x in range(len(data_arr) - max_number_of_days_per_example):
                    tar_data.extend([data_arr[x+max_number_of_days_per_example][0]])
                input_data.extend([[ticker, data_arr]])
                target_data.extend([[ticker, tar_data]])
            
            return self.__gen_training_examples(max_number_of_days_per_example, input_data, target_data)




class ProcessedDataHolder:
    
    def __init__(self, data_dictionaries):
        self.data = data_dictionaries
    
    def getTickerData(self, ticker):
        return self.data[ticker]
    
    def getAllData(self):
        ret = []
        for ticker, data in self.data.items():
            ret.append(data)
        return ret
    
    def getContained_tickers(self):
        return self.data.keys()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        