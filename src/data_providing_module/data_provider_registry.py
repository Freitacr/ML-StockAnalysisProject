import traceback
from abc import abstractmethod, ABC
from configparser import ConfigParser, SectionProxy
from typing import Dict, Tuple, List, Any
from general_utils.config.config_parser_singleton import read_execution_options
from general_utils.logging import logger


class Configurable (ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_configuration(self, section: "SectionProxy"):
        pass

    @abstractmethod
    def write_default_configuration(self, parser: "ConfigParser"):
        pass


class DataConsumerBase (Configurable, ABC):
    def __init__(self):
        super(DataConsumerBase, self).__init__()

    @abstractmethod
    def consumeData(self, data, passback, output_dir):
        pass

    @abstractmethod
    def predictData(self, data, passback, in_model_dir):
        pass


class DataProviderBase (Configurable, ABC):
    def __init__(self):
        super(DataProviderBase, self).__init__()

    @abstractmethod
    def generateData(self, login_credentials, *args, **kwargs):
        pass

    @abstractmethod
    def generatePredictionData(self, login_credentials, *args, **kwargs):
        pass


class DataProviderRegistry:

    def __init__(self):
        self.providers: Dict[str, "DataProviderBase"] = {}
        self.consumers: Dict[str, List[Tuple["DataConsumerBase", List[Any], Any]]] = {}

    def registerProvider(self, providerDataKey: str, provider: "DataProviderBase"):
        self.providers[providerDataKey] = provider

    def registerConsumer(self, providerDataKey: str, consumer, positional_arguments, passback=None):
        if providerDataKey not in self.consumers:
            self.consumers[providerDataKey] = []
        self.consumers[providerDataKey].append((consumer, positional_arguments, passback))

    def deregisterConsumer(self, providerDataKey: str, consumer):
        if providerDataKey not in self.consumers:
            return
        reg_consumers = self.consumers[providerDataKey]
        to_rem = []
        for consumer_list in reg_consumers:
            reg_consumer, _, _ = consumer_list
            if reg_consumer == consumer:
                to_rem.append(consumer_list)
        for obj in to_rem:
            self.consumers[providerDataKey].remove(obj)

    def deregisterProvider(self, providerDataKey: str):
        if providerDataKey in self.providers:
            self.providers.pop(providerDataKey)
        if providerDataKey in self.consumers:
            self.consumers.pop(providerDataKey)

    def passData(self, login_credentials, output_dir, stop_for_errors=False, print_errors=True):
        provider = None
        consumer = None
        columns = None
        try:
            predict = read_execution_options()
            ret_predictions = {}
            for provKey, provider in self.providers.items():
                if provKey not in self.consumers.keys():
                    continue
                registeredConsumers = self.consumers[provKey]
                for consumerSet in registeredConsumers:
                    consumer = None
                    args = None
                    passback = None
                    if len(consumerSet) == 3:
                        consumer, args, passback = consumerSet
                    elif len(consumerSet) == 2:
                        consumer, args = consumerSet
                    elif len(consumerSet) == 1:
                        consumer = consumerSet[0]
                    else:
                        raise ValueError("Invalid number of consumer registration arguments")

                    if not predict:
                        consumer.consumeData(provider.generateData(login_credentials, *args), passback, output_dir)
                    else:
                        ret_predictions[passback] = consumer.predictData(
                            provider.generatePredictionData(login_credentials, *args),
                            passback,
                            output_dir
                        )
            return ret_predictions
        except Exception:
            if print_errors:
                traceback.print_exc()
                logger.logger.log(logger.NON_FATAL_ERROR, "Above error was encountered during processing "
                                                          "of the following provider/consumer pair")
                logger.logger.log(
                    logger.NON_FATAL_ERROR,
                    "\t%s %s" % (type(provider), type(consumer))
                )
                logger.logger.log(logger.NON_FATAL_ERROR, "With the following columns as a data argument")
                logger.logger.log(logger.NON_FATAL_ERROR, "\t%s" % str(columns))
            if stop_for_errors:
                return


global registry
    
try:
    registry = registry
except NameError:
    registry = DataProviderRegistry()
