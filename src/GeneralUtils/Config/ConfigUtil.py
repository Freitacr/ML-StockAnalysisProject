from configparser import ConfigParser, SectionProxy
from DataProvidingModule.DataProviderRegistry import Configurable


def create_type_section(parser: "ConfigParser", configurable: "Configurable") -> SectionProxy:
    if not parser.has_section(str(type(configurable))):
        parser[str(type(configurable))] = {}
    return parser[str(type(configurable))]
