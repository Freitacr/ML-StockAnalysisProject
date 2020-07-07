import numpy as np
from typing import List, Sequence, Tuple


class MultiSequentialBlockEnvironment:

    def __init__(self):
        self.data: List[List["np.ndarray"]] = []
        self.step_index = 0
        self.reward_row = 2
        self.n_next_states = 2

    def reset(self) -> List["np.ndarray"]:
        self.step_index = 0
        return [np.array([x[self.step_index]]) for x in self.data]

    def setup_environment(self, data: List["np.ndarray"], stack_size: int = 5):
        # sets up the environment by stacking blocks together, the default stacking is 5 frames (a week of trading)
        self.data = [self.__split_block(x, stack_size) for x in data]

    def __split_block(self, data: "np.ndarray", stack_size: int) -> List["np.ndarray"]:
        ret: List["np.ndarray"] = []
        for i in range(data.shape[1] - stack_size + 1):
            ret.append(data[:, i:i+stack_size])
        return ret

    def step(self, actions: Sequence[int]) -> Tuple[
                                                    List["np.ndarray"],
                                                    List["float"],
                                                    bool
                                                ]:
        # todo update data splitting to stack frames properly
        if len(self.data) == 0:
            raise ValueError("No data found in the environment, did you forget to call setup_environment?")
        if not len(self.data) == len(actions):
            raise ValueError("Expected collection of %d ints, but received collection of %d ints" %
                             (len(self.data), len(actions)))
        if self.step_index == (len(self.data[0])-2):
            done = True
        else:
            done = False
        if not done:
            states = [np.array([x[self.step_index + 1]]) for x in self.data]
        else:
            states = [None] * len(self.data)

        rewards = [self.__calculate_reward(i, actions[i]) for i in range(len(self.data))]

        self.step_index += 1

        return states, rewards, done

    def __calculate_reward(self, state_list_index: int, action: int) -> float:
        # This is an uncertain decision, the thought process was to not incentivize not buying or selling so
        # the agent will tend toward buying or selling if it has reason to do so (aka in the past it sees the state
        # it is analyzing as being potentially good for one or the other) and if not (aka both buying and selling have
        # negative q scores, it will simply do nothing.

        if action == 0:
            return 0
        current_state = self.data[state_list_index][self.step_index]
        next_states = self.data[state_list_index][self.step_index + 1: self.step_index + self.n_next_states + 1]
        current_state = current_state[self.reward_row, -1]
        next_states = [x[self.reward_row, -1] - current_state for x in next_states]

        # if the step is to buy, then return the average of the differences calculated above
        # if it is to sell, then return the negated average of the differences above
        # the reasoning here is to give the model the goal of following average movement trends
        # if the average movement trend should be upward, then buy, and if it is downward then sell
        # These both map into stop shorting and start shorting as well

        if action == 1:
            return np.average(next_states)
        elif action == 2:
            return -np.average(next_states)
