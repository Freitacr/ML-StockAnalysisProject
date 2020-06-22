from DataProvidingModule.DataProviderRegistry import registry, DataProviderBase
from StockDataAnalysisModule.DataProcessingModule.StockClusterDataManager import StockClusterDataManager
from datetime import datetime as dt, timedelta as td


class SplitBlockProvider(DataProviderBase):

    def __init__(self):
        super(SplitBlockProvider, self).__init__()
        registry.registerProvider("SplitBlockProvider", self)

    def generateData(self, login_credentials, *args, **kwargs):
        if len(args) <= 1:
            raise ValueError('Expected at least the first argument from the following list;' +
                             ' train_columns: List["str"], expectation_columns: List["int"]')
        columns = args[0]
        expectation_columns = None
        if len(args) == 2:
            expectation_columns = args[1]
        start_date = dt.now() - td(weeks=(52 * 4))
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = dt.now().isoformat()[:10].replace('-', '/')
        data_retriever = StockClusterDataManager(login_credentials, start_date, end_date, column_list=columns)
        return data_retriever.retrieveTrainingDataMovementTargetsSplit(expectation_columns=expectation_columns)


try:
    provider = provider
except NameError:
    provider = SplitBlockProvider()
