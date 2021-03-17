import os
import importlib
from data_providing_module import configurable_registry
from data_providing_module.data_provider_registry import registry
from general_utils.config.config_parser_singleton import parser, update_config, read_execution_options
from general_utils.exportation import csv_amalgamation
from general_utils.logging import logger

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    providers = os.listdir("data_providing_module/data_providers")
    for provider in providers:
        if provider.startswith('__'):
            continue
        importlib.import_module('data_providing_module.data_providers.' + provider.replace('.py', ''))
    consumers = os.listdir("training_managers")
    for consumer in consumers:
        if consumer.startswith('__'):
            continue
        importlib.import_module("training_managers." + consumer.replace('.py', ''))
    configurable_registry.config_registry.handle_configurables(parser)
    predict, max_processes, export_predictions = read_execution_options()
    update_config()
    ret_predictions = registry.pass_data(args[0], stop_for_errors=False)
    if predict and not export_predictions:
        for passback, predictions in ret_predictions.items():
            logger.logger.log(logger.OUTPUT, predictions)
