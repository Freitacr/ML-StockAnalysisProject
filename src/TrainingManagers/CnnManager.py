from DataProvidingModule.DataProviderRegistry import registry, DataConsumerBase
from StockDataAnalysisModule.MLModels.KerasCnn import trainNetwork, createModel, evaluateNetwork


class CnnManager (DataConsumerBase):

    def __init__(self):
        super(CnnManager, self).__init__()
        registry.registerConsumer("ClusteredBlockProvider", self, [['hist_date', 'adj_close', 'opening_price'], [1]])
    
    def consumeData(self, data, passback):
        x_train, y_train, x_test, y_test = data
        data_shape = x_train[0].shape
        model = createModel(data_shape, num_out_categories=len(y_train[0]))
        trainNetwork(x_train, y_train, model)
        print(passback, evaluateNetwork(x_test, y_test, model))


try:
    consumer = consumer
except NameError:
    consumer = CnnManager()