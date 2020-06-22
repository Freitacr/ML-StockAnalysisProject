import traceback


class DataConsumerBase:
    def __init__(self):
        pass

    def consumeData(self, data, passback, output_dir):
        raise NotImplementedError()


class DataProviderBase:
    def __init__(self):
        pass

    def generateData(self, login_credentials, *args, **kwargs):
        raise NotImplementedError()


class DataProviderRegistry:

    def __init__(self):
        self.providers = {}
        self.consumers = {}

    def registerProvider(self, providerDataKey: str, provider : "DataProviderBase"):
        self.providers[providerDataKey] = provider

    def registerConsumer(self, providerDataKey: str, consumer, dataColumns, passback=None):
        if not providerDataKey in self.consumers:
            self.consumers[providerDataKey] = []
        self.consumers[providerDataKey].append([consumer, dataColumns, passback])

    def passData(self, login_credentials, output_dir, stop_for_errors=False, print_errors=True):
        provider = None
        consumer = None
        columns = None
        try:
            for provKey, provider in self.providers.items():
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
                    consumer.consumeData(provider.generateData(login_credentials, *args), passback, output_dir)
        except Exception:
            if print_errors:
                traceback.print_exc()
                print("Above error was encountered during processing of the following provider/consumer pair")
                print('\t', type(provider), type(consumer))
                print("With the following columns as a data argument:")
                print('\t', columns)
            if stop_for_errors:
                return


global registry
    
try:
    registry = registry
except NameError:
    registry = DataProviderRegistry()
