import configparser
from typing import Dict

from data_providing_module import data_provider_registry


class ConfigurableRegistry:

    def __init__(self):
        self._configurables: Dict[str, data_provider_registry.Configurable] = {}

    def register_configurable(self, configurable: data_provider_registry.Configurable):
        self._configurables[str(type(configurable))] = configurable

    def deregister_configurable(self, configurable: data_provider_registry.Configurable):
        if str(type(configurable)) in self._configurables:
            self._configurables.pop(str(type(configurable)))

    def handle_configurables(self, parser: configparser.ConfigParser):
        for _, configurable in self._configurables.items():
            configurable.load_configuration(parser)

config_registry = ConfigurableRegistry()
