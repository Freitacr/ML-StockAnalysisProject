from configparser import ConfigParser, SectionProxy
from general_utils.config import config_util as cfg_util
from data_providing_module.data_provider_registry import registry, DataProviderBase
from data_providing_module.data_providers import data_provider_static_names
from stock_data_analysis_module.data_processing_module.stock_cluster_data_manager import StockClusterDataManager
from datetime import datetime as dt, timedelta as td


ENABLED_CONFIG_ID = "enabled"


class ClusteredBlockProvider (DataProviderBase):
    
    def generatePredictionData(self, *args, **kwargs):
        pass

    def load_configuration(self, parser: "ConfigParser"):
        section = cfg_util.create_type_section(parser, self)
        if not parser.has_option(section.name, ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, ENABLED_CONFIG_ID)
        if not enabled:
            registry.deregisterProvider(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID)

    def write_default_configuration(self, section: "SectionProxy"):
        section[ENABLED_CONFIG_ID] = "True"

    def __init__(self):
        super(ClusteredBlockProvider, self).__init__()
        registry.registerProvider(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID, self)

    def generateData(self, *args, **kwargs):
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
        data_retriever = StockClusterDataManager(start_date, end_date, column_list=columns)
        return data_retriever.retrieveTrainingDataMovementTargets(expectation_columns=expectation_columns)


provider = ClusteredBlockProvider()
