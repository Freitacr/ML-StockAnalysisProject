from configparser import ConfigParser, NoSectionError, NoOptionError
from StockDataAnalysisModule.MLModels.KerasCnn import createModel, trainNetwork, evaluateNetwork
from StockDataAnalysisModule.DataProcessingModule.StockClusterDataManager import StockClusterDataManager
import numpy as np


def write_default_configs(parser, file_position):
    '''Writes the default configuration file at file_position'''
    parser.add_section('login_credentials')
    parser.set('login_credentials', 'user', 'root')
    parser.set('login_credentials', 'password', "")
    parser.set('login_credentials', 'database', 'stock_testing')
    parser.set('login_credentials', 'host', 'localhost')
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()


def config_handling():
    '''Handles reading in and error checking the configuration data'''
    file_position = "../configuration_data/config.ini"
    parser = ConfigParser()
    try:
        fp = open(file_position, 'r')
        fp.close()
    except FileNotFoundError:
        write_default_configs(parser, file_position)
    config_file = open(file_position, 'r')
    parser.read_file(config_file)
    try:
        user = parser.get('login_credentials', 'user')
        password = parser.get('login_credentials', 'password')
        database = parser.get('login_credentials', 'database')
        host = parser.get('login_credentials', 'host')
    except (NoSectionError, NoOptionError):
        write_default_configs(parser, file_position)
        user = parser.get('login_credentials', 'user')
        password = parser.get('login_credentials', 'password')
        database = parser.get('login_credentials', 'database')
        host = parser.get('login_credentials', 'host')
    return [host, user, password, database]


if __name__ == "__main__":
    login_credentials = config_handling()
    import sys
    args = sys.argv[1:]
    startDate = None
    endDate = None
    try:
        startDate = args[0]
        endDate = args[1]
    except IndexError:
        pass
    
    dataManager = StockClusterDataManager(login_credentials, startDate, endDate, num_similar_stocks=7)
    retMap = dataManager.retrieveTrainingDataMovementTargetsSplit(numDatsPerExample=30, normalized=False)
    for ticker, data in retMap.items():
        x_train, y_train, x_test, y_test = data
        temp = np.zeros((len(x_train), len(x_train[0]), len(x_train[0][0]), 1))
        for i in range(len(x_train)):
            for j in range(len(x_train[i])):
                for k in range(len(x_train[i][j])):
                    temp[i][j][k][0] = x_train[i][j][k]
        x_train = temp

        temp = np.zeros((len(x_test), len(x_test[0]), len(x_test[0][0]), 1))
        for i in range(len(x_test)):
            for j in range(len(x_test[i])):
                for k in range(len(x_test[i][j])):
                    temp[i][j][k][0] = x_test[i][j][k]
        x_test = temp
        
        cnn = createModel((8, 30, 1))
        bestAccuracy = 0
        bestCnn = None
        for i in range(200):
            cnn = trainNetwork(x_train, y_train, cnn)
            evalStats = evaluateNetwork(x_test, y_test, cnn) 
            print(evalStats)
            if evalStats[1] > bestAccuracy:
                bestAccuracy = evalStats[1]
                bestCnn = cnn
        
        bestCnn.save("bestcnn_%s.cmdl" % ticker)
