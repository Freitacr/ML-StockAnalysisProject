'''
Created on Dec 25, 2017

@author: Colton Freitas
'''

from .DataProcessingModule.DataProcessor import DataProcessor
from .MLModels import DecisionTree, NaiveBayes


#TODO: Create method for Decision Tree training
#TODO: Create class for info file managing (should be similar to configparser)



class MLManager:
    
    def __init__(self, login_credentials):
        self.data_handler = DataProcessor(login_credentials)
        self.valid_models = ["naive_bayes", "decision_tree"]
        
    def basicMovementsTraining(self, ML_Dir, ml_model = "naive_bayes"):
        if not ml_model.lower() in self.valid_models:
            raise ValueError("Invalid model %s\n Valid Models: %s" % (ml_model, str(self.valid_models)))
        
        ticker_data = self.data_handler.getMovementDirections(['adj_close', 'opening_price', 'close_price'], max_number_of_days_per_example = 30)
        stored_tickers = ticker_data.getContained_tickers()
        model_info_file = open(ML_Dir + "/%s_movement_direction_models.mlinf" % (ml_model), 'w')
        bigX, bigY = [[], []]
        for ticker in stored_tickers:
            X,Y = ticker_data.getTickerData(ticker)
            if ml_model.lower() == 'naive_bayes':
                ticker_bayes = NaiveBayes.NaiveBayes(assume_all_features_known=False)
            elif ml_model.lower() == 'decision_tree':
                ticker_bayes = DecisionTree.DecisionTree()
            ticker_bayes.train(X,Y)
            print("For %s" % ticker, self.__naiveBayesAccuracyTesting(ticker_bayes, ticker_data))
            bigX.extend(X)
            bigY.extend(Y)
        if ml_model == 'naive_bayes':
            big_bayes = NaiveBayes.NaiveBayes(assume_all_features_known=False)
        elif ml_model == 'decision_tree':
            big_bayes = DecisionTree.DecisionTree()
        big_bayes.train(bigX, bigY)
        print("For overall:", self.__naiveBayesAccuracyTesting(big_bayes, ticker_data))
        model_info_file.close()
        
    def __naiveBayesAccuracyTesting(self, bayes_model, ticker_data):
        ret_dict = {}
        bigX = []
        bigY = []
        for ticker in ticker_data.getContained_tickers():
            X,Y = ticker_data.getTickerData(ticker)
            correct = 0
            for example_index in range(len(X)):
                if str(bayes_model.predict(X[example_index])) == str(Y[example_index]):
                    correct += 1
            ret_dict[ticker] = float(correct) / len(X) #I cast to float because I'm used to Java, and don't trust integer division to be floating point return
            bigX.extend(X)
            bigY.extend(Y)
        correct = 0
        for example_index in range(len(bigX)):
            if str(bayes_model.predict(bigX[example_index])) == str(bigY[example_index]):
                correct += 1
        ret_dict['overall'] = float(correct) / len(bigX)
        return ret_dict
    
    
    
    
    
    
    
    