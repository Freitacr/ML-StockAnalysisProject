from typing import Tuple, Callable
from StockDataAnalysisModule.ReinforcementLearning.DeepQ.Memory.AgentTransitionMemoryABC import AgentTransitionMemoryABC
from StockDataAnalysisModule.ReinforcementLearning.DeepQ.Memory.CircularTransitionalMemory import CircularTransitionalMemory
from StockDataAnalysisModule.ReinforcementLearning.DeepQ.DeepQNetwork import DeepQNetwork
import numpy as np
from numpy import ndarray, argmax, arange, max as vmax
from numpy.random import random, choice
from os import path


class Agent:

    def __init__(self, input_shape: Tuple[int, int, int], learning_rate: float, n_actions: int,
                 future_reward_discount: float, replace_interval: int,
                 batch_size: int, random_action_chance: float = 1.0,
                 minimum_random_action_chance: float = .01,
                 transitional_memory: AgentTransitionMemoryABC = None, network_assembly_function: Callable = None,
                 network_fc1_dims: int = 512, random_action_chance_decay_factor: float = .999995):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.future_reward_discount = future_reward_discount
        self.replace_interval = replace_interval
        self.batch_size = batch_size
        self.random_action_chance = random_action_chance
        self.minimum_random_action_chance = minimum_random_action_chance
        self.memory = transitional_memory if transitional_memory is not None \
                                          else CircularTransitionalMemory(self.input_shape)
        network_assembly_args = [learning_rate, n_actions, input_shape, network_fc1_dims]
        if network_assembly_function is not None:
            network_assembly_args.append(network_assembly_function)
        self.q_next = DeepQNetwork(*network_assembly_args)
        self.q_eval = DeepQNetwork(*network_assembly_args)
        self.replace_counter = 0
        self.action_space = [i for i in range(n_actions)]
        self.random_action_chance_decay_factor = random_action_chance_decay_factor

    def choose_action(self, observed_state: "ndarray") -> int:
        if random() < self.random_action_chance:
            return choice(self.action_space)
        else:
            actions = self.predict_action_rewards(observed_state)
            return argmax(actions)

    def predict_action_rewards(self, observed_state: "ndarray") -> ndarray:
        state = np.array([observed_state], copy=False, dtype=np.float32)
        if np.array_equiv(state.shape, [1, 1]):
            print("problem")
        return self.q_eval.predict(state)

    def save(self, save_dir: str, include_optimizers: bool = True,
             q_next_filename: str = "qtarget.dq", q_eval_filename: str = "qeval.dq"):
        self.q_eval.store_model(path.join(save_dir, q_eval_filename), include_optimizers)
        self.q_next.store_model(path.join(save_dir, q_next_filename), include_optimizers)

    def load(self, save_dir: str,
             q_next_filename: str = "qtarget.dq", q_eval_filename: str = "qeval.dq"):
        self.q_eval.load_model(path.join(save_dir, q_eval_filename))
        self.q_next.load_model(path.join(save_dir, q_next_filename))

    def store_transition(self, state: "ndarray", action: int, reward: float, next_state: "ndarray", done: int):
        self.memory.store_transition(state, action, reward, next_state, done)

    def replace_target_network(self):
        if self.replace_interval is not None and self.replace_counter % self.replace_interval == 0:
            self.q_next.model.set_weights(self.q_eval.model.get_weights())

    def learn(self):
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, term_flags = self.memory.retrieve_transition_batch(self.batch_size)
            self.replace_target_network()

            pred_q_eval = self.q_eval.predict(states)
            pred_q_next = self.q_next.predict(next_states)

            q_target = pred_q_eval[:]
            indices = arange(self.batch_size)
            q_target[indices, actions] = rewards + (self.future_reward_discount *
                                                    vmax(pred_q_next, axis=1) * (1 - term_flags))
            self.q_eval.train(states, q_target)

            self.random_action_chance *= self.random_action_chance_decay_factor
            if self.random_action_chance < self.minimum_random_action_chance:
                self.random_action_chance = self.minimum_random_action_chance
            self.replace_counter += 1

        pass
