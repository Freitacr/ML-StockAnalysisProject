import sys
import os
import importlib
from DataProvidingModule.DataProviderRegistry import registry
from GeneralUtils.Config.ConfigParserSingleton import read_login_credentials, parser, \
    update_config, read_execution_options

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    host, user, database = read_login_credentials()
    providers = os.listdir("DataProvidingModule/DataProviders")
    for provider in providers:
        if provider.startswith('__'):
            continue
        importlib.import_module('DataProvidingModule.DataProviders.' + provider.replace('.py', ''))
    consumers = os.listdir("TrainingManagers")
    for consumer in consumers:
        if consumer.startswith('__'):
            continue
        importlib.import_module("TrainingManagers." + consumer.replace('.py', ''))
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
    ret_predictions = registry.passData([host, user, None, database], args[0], stop_for_errors=True)
    predict = read_execution_options()
    if predict:
        for passback, predictions in ret_predictions.items():
            for ticker, action_states in predictions.items():
                print("Predictions for %s" % ticker)
                for action, state in action_states:
                    print(action, state)
            # do something with the predictions
            pass
        pass
