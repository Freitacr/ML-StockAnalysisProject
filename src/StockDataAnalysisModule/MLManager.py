'''
Created on Dec 25, 2017

@author: Colton Freitas
'''

from .DataProcessingModule.DataProcessor import DataProcessor
from .MLModels import DecisionTree, NaiveBayes, KNN
from configparser import NoSectionError, NoOptionError
from .AnalysisUtils.ToleranceString import ToleranceString

#TODO: Use Tolerance String Class to make a tolerant model of accuracy. 

class MLManager:
    
    def __init__(self, login_credentials):
        self.data_handler = DataProcessor(login_credentials)
        self.valid_models = ["naive_bayes", "decision_tree"]
        
    def __getModelFromString(self, ml_model):
        if ml_model.lower() == 'naive_bayes':
            ret_model = NaiveBayes.NaiveBayes(assume_all_features_known=False)
        elif ml_model.lower() == 'decision_tree':
            ret_model = DecisionTree.DecisionTree()
        elif ml_model.lower() == 'knn':
            ret_model = KNN.KNN()
        return ret_model
    
    def __overallCrossfoldAccuracy(self, model, ticker_data, amount, numeric = False):
        tol_string = ToleranceString('extremes', amount, zero_state="None")
        ret_dict = {}
        stored_tickers = ticker_data.getContained_tickers()
        print("Started Cross-fold training")
        num_completed = 0
        for ticker in stored_tickers:
            num_completed += 1
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
                if not numeric:
                    if str(model.predict(testX[ex_index])) == str(testY[ex_index]):
                        correct += 1
                else:
                    if tol_string.test(float(model.predict(testX[ex_index])), float(testY[ex_index])):
                        correct += 1
            ret_dict [ticker] = correct / len(testX)
            print("Completed cross-fold training for ticker %s out of %s" % (num_completed, len(stored_tickers)))
        return ret_dict
    
    def __trainAndStoreModel(self, cross_model, ret_model, ticker_data, ML_Dir, methodName, ml_model, tolerant = False, amount = None, numeric = False):
        
        if tolerant and amount == None:
            raise ValueError("Tolerant set to True, but no amounts were specified")
        elif tolerant and not numeric or not tolerant and numeric:
            raise ValueError("Tolerant accuracy can only be used on numeric data")
        if not tolerant and numeric:
            amount = [0,0]
        info_man = InfoManager()
        stored_tickers = ticker_data.getContained_tickers()
        
        X,Y = [[], []]
        for ticker in stored_tickers:
            data = ticker_data.getTickerData(ticker)
            X.extend(data[0])
            Y.extend(data[1])
        print("Beginning Training...")
        ret_model.train(X, Y)
        print("Training completed.")
        
        info_man.addSectionWithOptions("overall", self.__overallCrossfoldAccuracy(cross_model, ticker_data, amount, numeric))
        #print("For overall:", self.__naiveBayesAccuracyTesting(big_bayes, ticker_data))
        info_man.writeFile(ML_Dir + "/%s_%s.mlinf" % (ml_model, methodName))
        ret_model.store(ML_Dir + "/%s_%s.mlmdl" % (ml_model, methodName))
    
    
    
    
    
    
    
    def basicMovementsTraining(self, ML_Dir, ml_model = "naive_bayes"):
        if not ml_model.lower() in self.valid_models:
            raise ValueError("Invalid model %s\nValid Models: %s" % (ml_model, str(self.valid_models)))
        
        ticker_data = self.data_handler.getMovementDirections(['adj_close', 'opening_price', 'close_price', 'volume_data'], max_number_of_days_per_example = 30)    
        cross_model = self.__getModelFromString(ml_model)
        ret_model = self.__getModelFromString(ml_model)
        self.__trainAndStoreModel(cross_model, ret_model, ticker_data, ML_Dir, "movement_direction_models", ml_model)
        
    def limitedNumericalChangeTraining(self, ML_Dir, ml_model = 'naive_bayes'):
        if not ml_model.lower() in self.valid_models:
            raise ValueError("Invalid model %s\nValid Models: %s" % (ml_model, str(self.valid_models)))
        
        ticker_data = self.data_handler.getLimitedNumericalChange(['adj_close', "opening_price", "close_price"], max_number_of_days_per_example = 30)
        cross_model = self.__getModelFromString(ml_model)
        ret_model = self.__getModelFromString(ml_model)
        self.__trainAndStoreModel(cross_model, ret_model, ticker_data, ML_Dir, 'Limited_Numerical_Change', ml_model, tolerant = True, amount = [10, 10], numeric = True)    
    
        
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    