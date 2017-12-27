'''
Created on Dec 25, 2017

@author: Colton Freitas
'''

from .DataProcessingModule.DataProcessor import DataProcessor
from .MLModels import DecisionTree, NaiveBayes, KNN
from configparser import NoSectionError, NoOptionError

class MLManager:
    
    def __init__(self, login_credentials):
        self.data_handler = DataProcessor(login_credentials)
        self.valid_models = ["naive_bayes", "decision_tree"]
        
    def basicMovementsTraining(self, ML_Dir, ml_model = "naive_bayes"):
        if not ml_model.lower() in self.valid_models:
            raise ValueError("Invalid model %s\nValid Models: %s" % (ml_model, str(self.valid_models)))
        info_man = InfoManager()
        ticker_data = self.data_handler.getMovementDirections(['adj_close', 'opening_price', 'close_price'], max_number_of_days_per_example = 30)
        stored_tickers = ticker_data.getContained_tickers()
        if ml_model.lower() == 'naive_bayes':
            cross_model = NaiveBayes.NaiveBayes(assume_all_features_known=False)
            ret_model = NaiveBayes.NaiveBayes(assume_all_features_known=False)
        elif ml_model.lower() == 'decision_tree':
            cross_model = DecisionTree.DecisionTree()
            ret_model = DecisionTree.DecisionTree()
        elif ml_model.lower() == 'knn':
            cross_model = KNN.KNN()
            ret_model = KNN.KNN()
        X,Y = [[], []]
        for ticker in stored_tickers:
            data = ticker_data.getTickerData(ticker)
            X.extend(data[0])
            Y.extend(data[1])
        print("Beginning Training...")
        ret_model.train(X, Y)
        print("Training completed.")
        
        info_man.addSectionWithOptions("overall", self.__overallCrossfoldAccuracy(cross_model, ticker_data))
        #print("For overall:", self.__naiveBayesAccuracyTesting(big_bayes, ticker_data))
        info_man.writeFile(ML_Dir + "/%s_movement_direction_models.mlinf" % (ml_model))
        ret_model.store(ML_Dir + "/%s_movement_direction_models.mlmdl" % (ml_model))
        
    def __overallCrossfoldAccuracy(self, model, ticker_data):
        ret_dict = {}
        stored_tickers = ticker_data.getContained_tickers()
        for ticker in stored_tickers:
            correct = 0
            testX, testY = ticker_data.getTickerData(ticker)
            X, Y = [[], []]
            for other_ticker in stored_tickers:
                if other_ticker == ticker:
                    continue
                oX, oY = ticker_data.getTickerData(other_ticker)
                X.extend(oX)
                Y.extend(oY)
            model.train(X,Y)
            for ex_index in range(len(testX)):
                if str(model.predict(testX[ex_index])) == str(testY[ex_index]):
                    correct += 1
            ret_dict [ticker] = correct / len(testX)
        return ret_dict
    
class InfoManager:
    
    def __init__(self):
        self.section_dict = {}
    
    def addSection(self, section_name):
        if not type(section_name) == type(""):
            raise ValueError("Section name was not of type %s, it was of type %s" % (str(type("")), str(type(section_name))))
        self.section_dict[section_name] = {}
    
    def addOption(self, section_name, option_name, value):
        try:
            self.section_dict[section_name][option_name] = value
        except KeyError:
            raise NoSectionError("No section named %s" % section_name)
    
    def writeFile(self, file_path):
        file = None
        try:
            file = open(file_path, 'w')
        except FileNotFoundError as e:
            raise e
        for key, value in self.section_dict.items():
            file.write("%s\n" % key)
            for sub_key, sub_value in value.items():
                file.write("\t%s:%s\n" % (sub_key, (sub_value)))
        file.close()
    
    def readFile(self, file_path):
        self.section_dict = {}
        file = None
        current_section = None
        try:
            file = open(file_path, 'r')
        except FileNotFoundError as e:
            raise e
        for line in file:
            if line[0] == "\t":
                linesplit = line.strip().split(":")
                self.addOption(current_section, linesplit[0], linesplit[1])
            else:
                self.addSection(line.strip())
                current_section = line.strip()
        file.close()
    
    def getOption(self, section, option):
        try:
            section = self.section_dict[section]
        except KeyError:
            raise NoSectionError("No section named %s" % section)
        try:
            return section[option]
        except KeyError:
            raise NoOptionError("No option named %s" % option)
    
    def getSectionList(self):
        return self.section_dict.keys()
    
    def getOptionList(self, section):
        try:
            return self.section_dict[section].keys()
        except KeyError:
            raise NoSectionError("No section named %s" % section)
        
    def addSectionWithOptions(self, section_name, options):
        if not type(section_name) == type(""):
            raise ValueError("Section name was not of type %s, it was of type %s" % (str(type("")), str(type(section_name))))
        self.section_dict[section_name] = options
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    