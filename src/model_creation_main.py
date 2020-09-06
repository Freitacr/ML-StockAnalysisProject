import os
import importlib
from data_providing_module import configurable_registry
from data_providing_module.data_provider_registry import registry
from general_utils.config.config_parser_singleton import parser, update_config, read_execution_options
from general_utils.logging import logger

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# suppress deprecation warnings in current thread
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# suppress Tensorflow warnings and info in current thread
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
    update_config()
    ret_predictions = registry.pass_data(args[0], stop_for_errors=False)
    predict, _ = read_execution_options()
    if predict:
        for passback, predictions in ret_predictions.items():
            logger.logger.log(logger.INFORMATION, predictions)
