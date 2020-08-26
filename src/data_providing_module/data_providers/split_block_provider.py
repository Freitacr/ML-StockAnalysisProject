"""Data Provider module for providing data blocks made from similar stocks over a set time period, but separated.

This data provider is not intended to be used outside of this module, instead, upon import, this module will create an
instance of a SplitBlockProvider and register it with the global DataProviderRegistry. To register a consumer to
receive data from this provider, use the id provided by data_provider_static_names.SPLIT_BLOCK_PROVIDER.

The separation, or split, referred to by this module is that the data block for one cluster is not combined with
the data block from others into a large training set. This is in contrast to the ClusteredBlockProvider, which
combines its cluster's blocks into a larger data set.

A detailed argument list that is required by this provider can be found in the generate_data method.
"""

from datetime import datetime as dt, timedelta as td
import configparser

from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names
from stock_data_analysis_module.data_processing_module import stock_cluster_data_manager
from general_utils.config import config_util


ENABLED_CONFIG_ID = "enabled"


class SplitBlockProvider(data_provider_registry.DataProviderBase):
    """Data Provider that provides data constructed by clustering stocks, but keeping the cluster's data separate

    The organization of these clusters is handled according to the specifications established in the
    StockClusterDataManager, and will operate on the time frame [start_date, end_date]. This time frame is currently
    fixed where end_date is the current date, and start_date is 52 * 4 weeks ago (approximately four years).

    Additionally this provider supports configuration of certain parameters through the configuration file. These
    parameters are listed in the Configurable Parameters section.

    Configurable Parameters:
        enabled: Whether this provider is enabled for consumers to receive data from.
    """

    def generate_prediction_data(self, *args, **kwargs):
        """Generates data that consumers will use to make predictions for the next trading day.

        Currently there is no implementation for this, and calling the method will result in a NotImplementedError
        """
        raise NotImplementedError()

    def __init__(self):
        """Initializes a SplitBlockProvider and registers it to the global DataProviderRegistry

        """
        super(SplitBlockProvider, self).__init__()
        data_provider_registry.registry.register_provider(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID, self)

    def write_default_configuration(self, section: "configparser.SectionProxy"):
        """Writes default configuration values into the SectionProxy provided.

        For more details see abstract class documentation.
        """
        section[ENABLED_CONFIG_ID] = "True"

    def load_configuration(self, parser: "configparser.ConfigParser"):
        """Attempts to load the configurable parameters for this provider from the provided parser.

        For more details see abstract class documentation.
        """
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, ENABLED_CONFIG_ID)
        if not enabled:
            data_provider_registry.registry.deregister_provider(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID)

    def generate_data(self, *args, **kwargs):
        """Generates data for Consumers to use by clustering together stocks in a time period,

        The time period for cluster creation is a period of 52 * 4 weeks (approximately 4 years).
        Consumers requiring data from this provider are expected to provide the arguments specified in the
        *args entry of the Arguments section

        The split portion of this data provider is that the data returned is split into different entries in a
        dictionary, keyed off of the root stock's ticker. The root stock is the stock that the cluster is based around
        and all other data in the cluster is deemed as being similar to the root stock's data.

        Arguments:
            *args:
                List of arguments that are expected to be in the following order, with the specified types
                train_columns: List[str]
                    List of names of columns from a StockDataTable. These will be used to retrieve data
                        from the database and construct the returned data blocks
                expectation_columns: List[int]
                    List of integers representing the indices of the columns to be used as the target data
                        in the generation of the data blocks
        Returns:
            See StockClusterDataManager.retrieve_training_data_movement_targets_split
        """
        if len(args) < 1:
            raise ValueError('Expected at least the first argument from the following list;' +
                             ' train_columns: List["str"], expectation_columns: List["int"]')
        columns = args[0]
        expectation_columns = None
        if len(args) == 2:
            expectation_columns = args[1]
        start_date = dt.now() - td(weeks=(52 * 4))
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = dt.now().isoformat()[:10].replace('-', '/')
        data_retriever = stock_cluster_data_manager.StockClusterDataManager(start_date, end_date, column_list=columns)
        return data_retriever.retrieveTrainingDataMovementTargetsSplit(expectation_columns=expectation_columns)


provider = SplitBlockProvider()
