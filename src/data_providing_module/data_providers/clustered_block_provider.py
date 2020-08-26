"""Data provider for data blocks made from similar stocks over a set time period

The data provider in this module is an implementation of a DataProviderBase, and is not intended to be instantiated
directly. Instead, upon import an instance of this provider will be created and registered with the global
DataProviderRegistry. From there, consumers can register with the registry with the id
data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID to receive data from this provider.

Detailed argument list that can be provided to this provider can be found in the generate_data method.
"""

import configparser
import datetime
from typing import Tuple

import numpy as np

from general_utils.config import config_util
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names
from stock_data_analysis_module.data_processing_module import stock_cluster_data_manager


_ENABLED_CONFIG_ID = "enabled"


class ClusteredBlockProvider (data_provider_registry.DataProviderBase):
    """Data Provider that will provide data constructed from organising similar stock's data into blocks.

    The organisation of these clusters is handled by a StockClusterDataManager, and will operate on the time frame
    [start_date, end_date], which are currently fixed where the end date is the current date, and the start date
    is 52 * 4 weeks ago.

    Additionally this provider provides access to configurable parameters through the config file. These parameters
    are listed in the Configurable Parameters section.

    Configurable Parameters:
        enabled: Whether this provider is enabled for consumers to receive data from
    """
    def generate_prediction_data(self, *args, **kwargs):
        """Generates data that consumers will use to make predictions for the next trading day.

        Currently there is no implementation for this, and calling this method will result in a NotImplementedError
        """
        raise NotImplementedError()

    def load_configuration(self, parser: "configparser.ConfigParser"):
        """Attempts to load the configurable parameters for this provider from the provided parser.

        For more details see abstract class documentation.
        """
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if not enabled:
            data_provider_registry.registry.deregister_provider(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID)

    def write_default_configuration(self, section: "configparser.SectionProxy"):
        """Writes default configuration values into the SectionProxy provided.

        For more details see abstract class documentation.
        """
        section[_ENABLED_CONFIG_ID] = "True"

    def __init__(self):
        """Initializes ClusteredBlockProvider and registers it with the global DataProviderRegistry

        """
        super(ClusteredBlockProvider, self).__init__()
        data_provider_registry.registry.register_provider(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID, self)

    def generate_data(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates data for Consumers to use by clustering together stocks in a time period

        The time period for cluster creation is a period of 52 * 4 weeks (approximately 4 years).
        Consumers requiring data from this provider are expected to provide the arguments specified in the
        *args entry of the Arguments section

        As a note, the data provided is not separated by cluster. If separation is desired, see SplitBlockProvider.

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
            See StockClusterDataManager.retrieve_training_data_movement_targets
        """
        if len(args) <= 1:
            raise ValueError('Expected at least the first argument from the following list;' +
                             ' train_columns: List["str"], expectation_columns: List["int"]')
        columns = args[0]
        expectation_columns = None
        if len(args) == 2:
            expectation_columns = args[1]
        start_date = datetime.datetime.now() - datetime.timedelta(weeks=(52 * 4))
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        data_retriever = stock_cluster_data_manager.StockClusterDataManager(start_date, end_date, column_list=columns)
        return data_retriever.retrieveTrainingDataMovementTargets(expectation_columns=expectation_columns)


provider = ClusteredBlockProvider()
