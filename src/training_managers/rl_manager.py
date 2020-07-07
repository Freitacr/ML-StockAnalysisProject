from configparser import ConfigParser, SectionProxy

from keras import Sequential

from general_utils.config import config_util as cfgUtil
from data_providing_module.data_provider_registry import DataConsumerBase, registry
from stock_data_analysis_module.reinforcement_learning.environment.multi_sequential_block_environment \
    import MultiSequentialBlockEnvironment
from stock_data_analysis_module.reinforcement_learning.deep_q.agent import Agent
from stock_data_analysis_module.reinforcement_learning.deep_q.memory.circular_transitional_memory \
    import CircularTransitionalMemory
from keras.layers import Conv2D, Dense, Flatten, Conv2DTranspose
from keras.optimizers import Adam
from typing import List
import numpy as np
from statistics import mean, stdev


ENABLED_CONFIGURATION_IDENTIFIER = "enabled"
LOAD_CHECKPOINT_CONFIGURATION_IDENTIFIER = "load_checkpoint"


def build_network(learning_rate, n_actions, input_shape, flatten_dense_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, strides=1, activation='relu',
                     input_shape=(*input_shape, ), data_format='channels_first'))
    model.add(Conv2D(filters=32, kernel_size=1, strides=1, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=2, strides=1, activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=2, strides=1, activation='relu',
                     data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(flatten_dense_dims, activation='relu'))
    model.add(Dense(n_actions))
    model.compile(optimizer=Adam(lr=learning_rate), loss="mean_squared_error")
    model.summary()
    return model


class RLManager (DataConsumerBase):


    def predictData(self, data, passback, in_model_dir):
        env = MultiSequentialBlockEnvironment()
        input_shape = (1, 8, 5)
        agent = Agent(input_shape, learning_rate=0.0, n_actions=3, network_assembly_function=build_network,
                      network_fc1_dims=64, random_action_chance=0.0, future_reward_discount=.99, replace_interval=1000,
                      batch_size=32)
        agent.load(in_model_dir, q_next_filename="multiblock-qtarget.dq", q_eval_filename="multiblock-qeval.dq")
        ret_predictions = {}
        for ticker, data_block, avg_price, avg_vol in data:
            env.setup_environment([data_block])
            current_state = env.reset()
            done = False
            action_states = []
            while not done:
                current_observation = current_state[0]
                action = agent.choose_action(current_observation)
                current_state, _, done = env.step([action])
                if done:
                    continue
                fixed_observation = np.zeros_like(current_observation[0])
                fixed_observation[:3, :] = current_observation[0, :3, :] * avg_price
                fixed_observation[3] = current_observation[0, 3] * avg_vol
                fixed_observation[4:7, :] = current_observation[0, 4:7, :] * avg_price
                action_states.append((action, fixed_observation))
            ret_predictions[ticker] = action_states
        return ret_predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = cfgUtil.create_type_section(parser, self)
        if not parser.has_option(section.name, ENABLED_CONFIGURATION_IDENTIFIER):
            self.write_default_configuration(section)
        if not parser.has_option(section.name, LOAD_CHECKPOINT_CONFIGURATION_IDENTIFIER):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, ENABLED_CONFIGURATION_IDENTIFIER)
        self.load_checkpoint = parser.getboolean(section.name, LOAD_CHECKPOINT_CONFIGURATION_IDENTIFIER)
        if not enabled:
            registry.deregisterConsumer("IndicatorBlockProvider", self)

    def write_default_configuration(self, section: "SectionProxy"):
        if ENABLED_CONFIGURATION_IDENTIFIER not in section.keys():
            section[ENABLED_CONFIGURATION_IDENTIFIER] = 'False'
        if LOAD_CHECKPOINT_CONFIGURATION_IDENTIFIER not in section.keys():
            section[LOAD_CHECKPOINT_CONFIGURATION_IDENTIFIER] = 'True'

    def __init__(self):
        super(RLManager, self).__init__()
        registry.registerConsumer("IndicatorBlockProvider", self, [260], passback="RL-Single")
        self.n_interations = 10000  # todo after configuration changes move this into a configuration file
        self.memory_size = 105000
        self.load_checkpoint = True

    def consumeData(self, data, passback, output_dir):
        print("... training using RLManager ...")
        env = MultiSequentialBlockEnvironment()
        input_shape = (1, 8, 5)
        agent_memory = CircularTransitionalMemory(input_shape, self.memory_size)
        agent_random_action_chance_initial = 1.0 if not self.load_checkpoint else .3
        agent = Agent(input_shape, learning_rate=1e-5, n_actions=3, future_reward_discount=.99,
                      replace_interval=1000, batch_size=256, transitional_memory=agent_memory,
                      network_assembly_function=build_network, network_fc1_dims=128,
                      random_action_chance=agent_random_action_chance_initial,
                      random_action_chance_decay_factor=.99999993)
        if self.load_checkpoint:
            agent.load(output_dir, q_next_filename="multiblock-qtarget.dq", q_eval_filename="multiblock-qeval.dq")
        best_potential = 0
        env.setup_environment(data)
        learning_index = 0
        for i in range(self.n_interations):
            curr_step_index = 0
            current_observations = env.reset()
            done = False
            potentials = []
            while not done:
                curr_step_index += 1
                actions = generate_actions(agent, current_observations)
                next_observations, rewards, done = env.step(actions)
                potentials.extend(rewards)
                for observation_index in range(len(actions)):
                    agent.store_transition(
                        current_observations[observation_index],
                        actions[observation_index],
                        rewards[observation_index],
                        next_observations[observation_index],
                        1 if done else 0
                    )
                    learning_index += 1
                    if learning_index % 10 == 0:
                        agent.learn()
                        learning_index = 0
                current_observations = next_observations
                # if curr_step_index % 10 == 0:
                #     print("Current step index: %d" % curr_step_index)
            avg_potential = np.average(potentials)
            print("episode %d:\naverage potential: %.6f\nepsilon: %.2f\nmin_potential: %.6f\nmax_potential: %.6f\n"
                  "std potentials: %.8f" %
                  (i, avg_potential, agent.random_action_chance,
                   np.min(potentials), np.max(potentials), np.std(potentials)))
            if avg_potential > best_potential:
                print("average potential of %.6f is better than best potential of %.6f; saving models" %
                      (avg_potential, best_potential))
                best_potential = avg_potential
                agent.save(output_dir, q_next_filename="multiblock-qtarget.dq", q_eval_filename="multiblock-qeval.dq")
        print("... RLManager managed training finished ...")
        pass


def generate_actions(agent: "Agent", observations: List["np.ndarray"]):
    ret_actions = [agent.choose_action(obs) for obs in observations]
    return ret_actions


try:
    provider = provider
except NameError:
    provider = RLManager()
