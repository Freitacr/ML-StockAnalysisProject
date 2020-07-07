from StockDataAnalysisModule.ReinforcementLearning.DeepQ.Memory.AgentTransitionMemoryABC import AgentTransitionMemoryABC
from typing import Tuple, List
import numpy as np


class CircularTransitionalMemory(AgentTransitionMemoryABC):

    def __init__(self, input_shape: Tuple[int, int, int], memory_size: int = 300):
        super(CircularTransitionalMemory, self).__init__()
        self.state_mem: "np.ndarray" = np.zeros((memory_size, *input_shape), dtype=np.float32)
        self.action_mem: "np.ndarray" = np.zeros((memory_size, ), dtype=np.int32)
        self.new_state_mem: "np.ndarray" = np.zeros((memory_size, *input_shape), dtype=np.float32)
        self.reward_mem: "np.ndarray" = np.zeros((memory_size, ), dtype=np.float32)
        self.term_mem: "np.ndarray" = np.zeros((memory_size, ), dtype=np.uint8)
        self.curr_index = 0
        self.memory_size = memory_size

    def store_transition(self, state: "np.ndarray", action: int, reward: float, new_state: "np.ndarray", done: int):
        # store at current index in all memory arrays, update index and mod to keep in bound
        mem_index = self.curr_index % self.memory_size
        self.state_mem[mem_index] = state
        self.action_mem[mem_index] = action
        self.reward_mem[mem_index] = reward
        self.new_state_mem[mem_index] = new_state
        self.term_mem[mem_index] = done
        self.curr_index += 1

    def retrieve_transition_batch(self, batch_size: int) -> Tuple[
                                                                "np.ndarray",
                                                                "np.ndarray",
                                                                "np.ndarray",
                                                                "np.ndarray",
                                                                "np.ndarray"
                                                            ]:
        if batch_size > len(self):
            raise ValueError("Not enough transitions to retrieve batch of size %d. Only %d are available"
                             % (batch_size, len(self)))
        available_memory = len(self)
        chosen_indices = np.random.choice(available_memory, batch_size, replace=False)
        states = self.state_mem[chosen_indices]
        actions = self.action_mem[chosen_indices]
        rewards = self.reward_mem[chosen_indices]
        new_states = self.new_state_mem[chosen_indices]
        term_memory = self.term_mem[chosen_indices]
        return states, actions, rewards, new_states, term_memory

    def __len__(self):
        return min(self.curr_index, self.memory_size)
