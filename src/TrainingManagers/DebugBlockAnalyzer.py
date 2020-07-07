from configparser import ConfigParser, SectionProxy
from GeneralUtils.Config import ConfigUtil as cfgUtil
from DataProvidingModule.DataProviderRegistry import DataConsumerBase, registry


class DebugBlockAnalyzer (DataConsumerBase):

    def predictData(self, data, passback, in_model_dir):
        pass

    def load_configuration(self, parser: "ConfigParser"):
        section = cfgUtil.create_type_section(parser, self)
        if not parser.has_option(section.name, "enabled"):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, 'enabled')
        if not enabled:
            registry.deregisterConsumer("IndicatorBlockProvider", self)
        else:
            registry.consumers.clear()
            registry.registerConsumer("IndicatorBlockProvider", self,
                                      [75], passback="debug")

    def write_default_configuration(self, section: "SectionProxy"):
        section['enabled'] = 'False'

    def __init__(self):
        super(DebugBlockAnalyzer, self).__init__()
        registry.registerConsumer("IndicatorBlockProvider", self,
                                  [75], passback="debug")

    def consumeData(self, data, passback, output_dir):
        print(data)
        pass


try:
    consumer = consumer
except NameError:
    consumer = DebugBlockAnalyzer()
