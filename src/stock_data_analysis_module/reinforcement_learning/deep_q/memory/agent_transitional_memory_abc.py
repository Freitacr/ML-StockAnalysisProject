from abc import ABC, abstractmethod
import numpy as np


class AgentTransitionMemoryABC(ABC):

    @abstractmethod
    def store_transition(self, state: "np.ndarray", action: "int", reward: float, new_state: "np.ndarray", done: int):
        pass

    @abstractmethod
    def retrieve_transition_batch(self, batch_size: int):
        pass

    @abstractmethod
    def __len__(self):
        pass