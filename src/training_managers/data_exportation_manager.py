"""Module for direct exportation of generated data into static files

The training manager in this module is intended to be used as a
tool to generate a fixed dataset for doing model experimentation on
without worrying about database access and creation of a new
training manager.
"""
from configparser import ConfigParser, SectionProxy
from os import path

import pickle

from general_utils.config import config_util
from data_providing_module.data_provider_registry import DataConsumerBase
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names
from general_utils.mysql_management.mysql_tables import stock_data_table


_ENABLED_CONFIGURATION_IDENTIFIER = "enabled"
CONSUMER_IDENTIFIER = "DataExportationManager"


class DataExportationManager(DataConsumerBase):

    def consume_data(self, data, passback, output_dir):
        out_file_path = output_dir + path.sep + passback + '.pickle'
        with open(out_file_path, "wb") as out_file:
            pickle.dump(data, out_file)

    def predict_data(self, data, passback, in_model_dir):
        """Method does nothing as this class does not produce predictions"""
        return None

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIGURATION_IDENTIFIER):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        if not enabled:
            data_provider_registry.registry.deregister_consumer(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID,
                                                                self)
            data_provider_registry.registry.deregister_consumer(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID,
                                                                self)
            data_provider_registry.registry.deregister_consumer(data_provider_static_names.INDICATOR_BLOCK_PROVIDER_ID,
                                                                self)
            data_provider_registry.registry.deregister_consumer(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                self)

    def __init__(self):
        super(DataExportationManager, self).__init__()
        data_columns = [stock_data_table.HIGH_PRICE_COLUMN_NAME, stock_data_table.HIGH_PRICE_COLUMN_NAME,
                        stock_data_table.LOW_PRICE_COLUMN_NAME, stock_data_table.VOLUME_COLUMN_NAME,
                        stock_data_table.CLOSING_PRICE_COLUMN_NAME]
        data_provider_registry.registry.register_consumer(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID,
                                                          self, [data_columns, [7]],
                                                          data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID)
        data_provider_registry.registry.register_consumer(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID,
                                                          self, [data_columns, [7]],
                                                          data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID)
        data_provider_registry.registry.register_consumer(data_provider_static_names.INDICATOR_BLOCK_PROVIDER_ID,
                                                          self, [200],
                                                          data_provider_static_names.INDICATOR_BLOCK_PROVIDER_ID)
        data_provider_registry.registry.register_consumer(
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
            self, [252 * 10],
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID)

    def write_default_configuration(self, section: "SectionProxy"):
        if _ENABLED_CONFIGURATION_IDENTIFIER not in section.keys():
            section[_ENABLED_CONFIGURATION_IDENTIFIER] = 'False'


consumer = DataExportationManager()
