from typing import Any
from serializable import Serializable

class Plugin(Serializable):

    def __init__(self) -> None:
        pass

    def step(self, attached: Any) -> Any:
        pass

    def reset(self, attached: Any) -> Any:
        pass

class Pluggable(Serializable):

    def __init__(self) -> None:
        self.plugins : list[Plugin] = []

    def step(self) -> None:
        for plugin in self.plugins:
            plugin.step(self)
    
    def reset(self) -> None:
        for plugin in self.plugins:
            plugin.reset(self)
    
    def plug(self, plugin: Plugin):
        self.plugins.append(plugin)
    