from data_providing_module.data_provider_registry import registry, DataProviderBase
from stock_data_analysis_module.data_processing_module.stock_cluster_data_manager import StockClusterDataManager
from datetime import datetime as dt, timedelta as td
from configparser import ConfigParser, SectionProxy
from general_utils.config import config_util as cfgUtil


ENABLED_CONFIG_ID = "enabled"


class SplitBlockProvider(DataProviderBase):

    def generatePredictionData(self, login_credentials, *args, **kwargs):
        pass

    def __init__(self):
        super(SplitBlockProvider, self).__init__()
        registry.registerProvider("SplitBlockProvider", self)

    def write_default_configuration(self, section: "SectionProxy"):
        section[ENABLED_CONFIG_ID] = "True"

    def load_configuration(self, parser: "ConfigParser"):
        section = cfgUtil.create_type_section(parser, self)
        if not parser.has_option(section.name, ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, ENABLED_CONFIG_ID)
        if not enabled:
            registry.deregisterProvider("SplitBlockProvider")

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
