"""Module built to create a registry for Data Providers and Data Consumers.

Module Contents:
    Configurable:
        Abstract class representing an object that supports loading configurations from a ConfigParser object.
    DataProviderBase:
        Abstract class representing an object capable of generating data for consumers.
    DataConsumerBase:
        Abstract class representing an object capable using data generated by a provider.
    DataProviderRegistry:
        Registry allowing for abstract interaction between DataConsumers and DataProviders. Providers register
        themselves using an id, and consumers register themselves by the id of the provider they want to request data
        from. The interactions between the two are performed in the execution of the pass_data method.

    registry:
        DataProviderRegistry that acts as an importable, shared registry. The contents of the registry should not
        be modified directly without a strong reason.

        Usage Examples:
            Registering a Provider with id SomeDataProvider:
                registry.register_provider("SomeDataProvider", provider_object)

            Registering a Consumer to receive data from provider with id SomeDataProvider:
                assume that pos_args is a list of positional arguments that SomeDataProvider needs to generate data
                registry.register_consumer("SomeDataProvider", consumer_object, pos_args)

            Initializing the passing of information from Providers to their respective registered Consumers:
                Assume that out_dir is a string representing the path to a directory where files can be written by
                    consumers
                registry.pass_data(out_dir, stop_for_errors=True, print_errors=True)
"""
import traceback
from abc import abstractmethod, ABC
from configparser import ConfigParser, SectionProxy
from typing import Dict, Tuple, List, Any
from general_utils.config.config_parser_singleton import read_execution_options
from general_utils.logging import logger


class Configurable (ABC):
    """Abstract class representing an object that supports loading configurations from a ConfigParser object.

    """
    def __init__(self):
        pass

    @abstractmethod
    def load_configuration(self, parser: "ConfigParser"):
        """Attempts to load the configurable parameters for this provider from the provided parser.

        In the event that the parser does not have a section for this provider (usually because no configuration file
        has been written), then default parameters should be written through the write_default_configuration method.

        Arguments:
                parser: ConfigParser object that has already read the configuration file
        """
        pass

    @abstractmethod
    def write_default_configuration(self, section: "SectionProxy"):
        """Writes default configuration values into the SectionProxy provided.

        In the event that loading a configuration did not yield a section for this provider, or the configurations
        need to be reset to their defaults, this method should write the default configuration parameters for this
        provider into the SectionProxy.

        Arguments:
            section: SectionProxy representing the section within a ConfigParser with the name resulting from
                str(type(self))
        """
        pass


class DataConsumerBase (Configurable, ABC):
    """Abstract class representing an object capable using data generated by a provider.

    """
    def __init__(self):
        super(DataConsumerBase, self).__init__()

    @abstractmethod
    def consume_data(self, data, passback, output_dir):
        """Consume data generated from a DataProvider

        A Consumer implementing this method is expected to receive data generated by a DataProvider for the purposes
        of training models to eventually make predictions about stock movements. Trained model files are expected
        to be stored in output_dir, or in a subdirectory thereof.

        Arguments:
            data: Any
                The data passed on by a data provider that the consumer registered to receive data from
            passback: Any
                Object registered along side the Consumer in the DataProviderRegistry. This is intended
                to ease differentiation between which Provider is providing the data in the data argument. If this
                consumer only registered to receive data from one provider, such a passback value is not necessary.
            output_dir: str
                String path to the directory that trained model files should be stored in. Consumers are allowed
                to further create and maintain subdirectories if they expect model name clashes in the base output
                directory.
        Returns:
            None. This method should not return any values.
        """
        pass

    @abstractmethod
    def predict_data(self, data, passback, in_model_dir):
        """Predict the next day's state using the data provided from a Data Provider.

        A Consumer implementing this method is expected to receive data generated by a DataProvider for the purposes
        of making predictions about stock movements in the future. How these predictions are structured and returned
        are at the discretion of the implementing Consumer.

        Arguments:
            data: Any
                The data passed on by a data provider that the consumer registered to receive data from
            passback: Any
                Object registered along side the Consumer in the DataProviderRegistry. This is intended
                to ease differentiation between which Provider is providing the data in the data argument. If this
                consumer only registered to receive data from one provider, such a passback value is not necessary.
            in_model_dir: str
                String path to the directory that trained model files should already be stored in. If the Consumer
                has created further sub directories to avoid model naming conflicts, they are expected to load files
                from that position.
        Returns:
            This abstract definition does not restrict what predictions a Consumer is required to make, nor how they
            are returned.
            TODO[Colton Freitas] After global rewrite, define a new abstract method for converting predictions into
                a human readable format.
        """
        pass


class DataProviderBase (Configurable, ABC):
    """Abstract class representing an object capable of generating data for consumers.


    """
    def __init__(self):
        super(DataProviderBase, self).__init__()

    @abstractmethod
    def generate_data(self, *args, **kwargs):
        """Generate data to satisfy a request from a DataConsumer

        A Provider implementing this method is expected to generate data for a DataConsumer, using the arguments
        that consumer has provided through the DataProviderRegistry.
        TODO[Colton Freitas] Update current DataProviders to provide training method agnostic data, and move the
            responsibility for structuring the data into the format required for the training method to the DataConsumer
            Classes. Do so after the global code refactoring is completed.

        Arguments:
            Arguments for this method are defined by the implementing class.

        Returns:
            Returns are currently defined by the implementing Provider. As noted in the method description, this is
            soon to change to a training method agnostic return style.
        """
        pass

    @abstractmethod
    def generate_prediction_data(self, *args, **kwargs):
        """Generate data to satisfy a request from a DataConsumer with additional details to make prediction manageable

        A provider implementing this data is expected to generate all data required for a DataConsumer to make
        predictions regarding the state of the watched stock list for the following day. Currently this level of data
        required has not been standardized

        TODO[Colton Freitas] Update current DataProviders to provide training method agnostic data for predictions,
            and move the responsibility for structuring that data into the format required for the particular training
            method to the DataConsumer classes. This should be begun after the global code refactoring is complete.

        Arguments:
            Arguments for this method are defined by the implementing class.

        Returns:
            Returns are currently defined by the implementing Provider. As noted in the method description, this is
            soon to change to a training method agnostic return style.
        """
        pass


class DataProviderRegistry:
    """Registry allowing for abstract interaction between DataConsumers and DataProviders.

    Registry allowing for abstract interaction between DataConsumers and DataProviders. Providers register
    themselves using an id, and consumers register themselves by the id of the provider they want to request data
    from. The interactions between the two are performed in the execution of the pass_data method.
    
    """

    def __init__(self):
        self.providers: Dict[str, "DataProviderBase"] = {}
        self.consumers: Dict[str, List[Tuple["DataConsumerBase", List[Any], Any]]] = {}

    def register_provider(self, providerDataKey: str, provider: "DataProviderBase"):
        self.providers[providerDataKey] = provider

    def register_consumer(self, providerDataKey: str, consumer, positional_arguments, passback=None):
        if providerDataKey not in self.consumers:
            self.consumers[providerDataKey] = []
        self.consumers[providerDataKey].append((consumer, positional_arguments, passback))

    def deregister_consumer(self, providerDataKey: str, consumer):
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

    def deregister_provider(self, providerDataKey: str):
        if providerDataKey in self.providers:
            self.providers.pop(providerDataKey)
        if providerDataKey in self.consumers:
            self.consumers.pop(providerDataKey)

    def pass_data(self, output_dir, stop_for_errors=False, print_errors=True):
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
                        consumer.consume_data(provider.generate_data(*args), passback, output_dir)
                    else:
                        ret_predictions[passback] = consumer.predict_data(
                            provider.generate_prediction_data(*args),
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


registry = DataProviderRegistry()
