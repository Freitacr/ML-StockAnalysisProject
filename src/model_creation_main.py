import os
import importlib
from data_providing_module.data_provider_registry import registry
from general_utils.config.config_parser_singleton import parser, update_config, read_execution_options
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
    providers = list(registry.providers.values())
    for provider in providers:
        provider.load_configuration(parser)
    consumers = list(registry.consumers.values())
    for i in range(len(consumers)):
        consumers[i] = consumers[i][:]
    for consumer_list in consumers:
        for consumer, _, _ in consumer_list:
            consumer.load_configuration(parser)
    update_config()
    ret_predictions = registry.pass_data(args[0], stop_for_errors=False)
    predict = read_execution_options()
    if predict:
        for passback, predictions in ret_predictions.items():
            for ticker, prediction in predictions.items():
                logger.logger.log(logger.INFORMATION, "Predictions for %s" % ticker)
                predicted_direction, observed_accuracy = prediction
                logger.logger.log(
                    logger.INFORMATION,
                    "%s was theorized with an observed accuracy of %s" %
                    (str(predicted_direction), str(observed_accuracy))
                )
