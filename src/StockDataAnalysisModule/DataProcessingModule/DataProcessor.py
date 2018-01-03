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
    
    def getPercentageChanges(self, column_list, max_number_of_days_per_example = 5):
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
                #This is true because there is nothing to compare the first day in the stock's history against but itself, so the change HAS to be zero.
                data[0][data_type_index] = 0
            
        
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
        
    def __gen_training_examples(self, max_number_of_days_per_example, input_data, output_data):
        #    
        if not (len(input_data[0][1])) - max_number_of_days_per_example == len(output_data[0][1]):
            raise InvalidStateError("Input size mismatch: expected %s, but got %s" % ((len(input_data[0][1])) - max_number_of_days_per_example, len(output_data[0][1])))
        #assumed input data in format [ [ticker, in_data] ... ]
        #assumed output data in format [ [ticker, out_data] ... ]
        ret_dict = {}
        for ticker_index in range(len(input_data)):
            X, Y = [[], []]
            in_data = input_data[ticker_index][1]
            class_data = output_data[ticker_index][1]
            for example_index in range(len(in_data) - max_number_of_days_per_example):
                if not type(in_data[example_index]) == type([]):
                    #If values in in_data are not lists
                    X.extend(in_data[example_index:example_index + max_number_of_days_per_example])
                else:
                    full_list = []
                    for sublist in in_data[example_index:example_index + max_number_of_days_per_example]:
                        full_list.extend(sublist)
                    X.extend([full_list])
                Y.extend([class_data[example_index]])
            ret_dict[input_data[ticker_index][0]] = [X, Y]
        return ProcessedDataHolder(ret_dict)   
    
     
    def getMovementDirections(self, column_list, max_number_of_days_per_example = 5):
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
                #in_data = []
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        