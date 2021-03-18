from typing import Optional, List, Tuple, IO

import numpy as np
import pickle


class TradingPopulation:

    def __init__(self, input_shape: Tuple[int, int], starting_balance: float, num_individuals: int,
                 mutation_chance_genome=.1, mutation_magnitude=.15, crossover_chance_genome=.5):
        self.__num_individuals = num_individuals
        self.__best_individual: Optional[TradingIndividual] = None
        self.__contained_individuals: List[TradingIndividual] = []
        self.__input_shape = input_shape
        self.__starting_balance = starting_balance
        self.__mutation_chance = mutation_chance_genome
        self.__mutation_magnitude = mutation_magnitude
        self.__crossover_chance = crossover_chance_genome

    def save(self, file_name: str):
        with open(file_name, 'wb') as open_handle:
            open_handle.write(str(self.__num_individuals).encode(encoding="UTF8") + b'\n')
            pickle.dump(self.__input_shape, open_handle)
            for individual in self.__contained_individuals:
                individual.save(open_handle)

    def load(self, file_name: str):
        with open(file_name, 'rb') as open_handle:
            self.__contained_individuals = []
            num_individuals = int(open_handle.readline())
            self.__num_individuals = num_individuals
            self.__input_shape = pickle.load(open_handle)
            for i in range(num_individuals):
                self.__contained_individuals.append(TradingIndividual((1, 1), 0))
                self.__contained_individuals[-1].load(open_handle)

    def train(self, input_data: np.ndarray, epochs: int, share_prices: List[float]) -> List[float]:
        # note that input_data in this context is a Kxmxn matrix. Where K is the number of examples in the dataset for
        # one stock. It is also assumed that the each mxn matrix is sequentially in order.
        # Meaning that the example paired with July 14 is after the one for July 13, and before the one for July 15.
        if len(self.__contained_individuals) == 0:
            self.__spawn_remaining_population()
        best_fitness = []
        for i in range(epochs):
            for j in range(len(input_data)):
                example_data = input_data[j]
                share_price = share_prices[j]
                self.__epoch_iteration(example_data, share_price)
            best_fitness = self.__generate_next_generation()
        return best_fitness

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        ret_predictions = np.zeros((3, 2))
        for i in range(3):
            ret_predictions[i] = self.__contained_individuals[i].predict_data(input_data)
        return ret_predictions

    def __epoch_iteration(self, input_data: np.ndarray, share_price: float):
        for individual in self.__contained_individuals:
            individual.handle_data(input_data, share_price)

    def __generate_next_generation(self):
        self.__contained_individuals = sorted(self.__contained_individuals,
                                              key=lambda x: x.calculate_fitness(),
                                              reverse=True)
        mutate_pop: List[TradingIndividual] = []
        crossover_pop: List[TradingIndividual] = []
        kept_pop = self.__contained_individuals[:round(.1 * self.__num_individuals)]
        best_fitness = [x.calculate_fitness() for x in kept_pop[:3]]
        for individual in kept_pop:
            individual.reset_starting_state()
        for i in range(round(.3 * self.__num_individuals)):
            selected_individual = kept_pop[round(np.random.ranf() * len(kept_pop)) - 1]
            mutate_pop.append(selected_individual.mutate(self.__mutation_chance, self.__mutation_magnitude))
        for i in range(round(.2 * self.__num_individuals)):
            selected_individual_a = kept_pop[round(np.random.ranf() * len(kept_pop)) - 1]
            selected_individual_b = kept_pop[round(np.random.ranf() * len(kept_pop)) - 1]
            while selected_individual_a == selected_individual_b:
                selected_individual_b = kept_pop[round(np.random.ranf() * len(kept_pop)) - 1]
            crossover_pop.append(selected_individual_a.crossover(selected_individual_b, self.__crossover_chance))
        kept_pop.extend(mutate_pop)
        kept_pop.extend(crossover_pop)
        self.__contained_individuals = kept_pop
        self.__spawn_remaining_population()
        return best_fitness

    def __spawn_remaining_population(self):
        for i in range(self.__num_individuals - len(self.__contained_individuals)):
            self.__contained_individuals.append(TradingIndividual(self.__input_shape, self.__starting_balance))


class TradingIndividual:

    def __init__(self, input_shape: Tuple[int, int], starting_balance: float):
        self.__days_share_held = 0
        self.__current_balance = starting_balance
        self.__starting_balance = starting_balance
        self.__transaction_state = False
        self.__held_share_price = 0.0
        self.__num_held_shares = 0
        self.__buy_state_matrices: List[np.ndarray] = []
        self.__sell_state_matrices: List[np.ndarray] = []
        self.__initialize_transaction_matrices(input_shape)
        self.__input_shape = input_shape

    def __initialize_transaction_matrices(self, input_shape: Tuple[int, int]):
        # Initialize transaction matrices with values in the range [-2, 2)
        # With this range, values should neither explode nor degrade during multiplications too badly.

        def shift_matrix_factory(shape):
            return np.full(shape, 2)

        self.__buy_state_matrices.clear()
        self.__sell_state_matrices.clear()
        reframe_shape = (input_shape[1], input_shape[0])
        analysis_shape = (input_shape[0], input_shape[0])
        weighing_shape = (input_shape[0], 2)
        conclusory_shape = (1, input_shape[0])
        self.__buy_state_matrices.append((np.random.ranf(reframe_shape) * 4) - shift_matrix_factory(reframe_shape))
        self.__buy_state_matrices.append((np.random.ranf(analysis_shape) * 4) - shift_matrix_factory(analysis_shape))
        self.__buy_state_matrices.append((np.random.ranf(weighing_shape) * 4) - shift_matrix_factory(weighing_shape))
        self.__buy_state_matrices.append(
            (np.random.ranf(conclusory_shape) * 4) - shift_matrix_factory(conclusory_shape))
        self.__sell_state_matrices.append((np.random.ranf(reframe_shape) * 4) - shift_matrix_factory(reframe_shape))
        self.__sell_state_matrices.append((np.random.ranf(analysis_shape) * 4) - shift_matrix_factory(analysis_shape))
        self.__sell_state_matrices.append((np.random.ranf(weighing_shape) * 4) - shift_matrix_factory(weighing_shape))
        self.__sell_state_matrices.append(
            (np.random.ranf(conclusory_shape) * 4) - shift_matrix_factory(conclusory_shape))

    def __mutate_matrix(self, chance_per_genome: float, rate_per_selected: float, matrix: np.ndarray) -> np.ndarray:
        # The current decision is to base the mutation of each genome on the current strength of the genome.
        # This may prevent explosion of values and will be more precise at lower magnitude numbers.
        # This comes at the cost of a loss of precise mutations with higher magnitude numbers.
        # The other option is to set the maximum magnitude of mutation, and have the rate be a multiplier on that.
        ret_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                genome_chance = np.random.ranf()
                if genome_chance < chance_per_genome:
                    genome_chance = np.random.ranf()
                    mutation_magnitude = matrix[i][j] * rate_per_selected
                    if genome_chance < .5:
                        mutation_magnitude *= -1
                    ret_matrix[i][j] = matrix[i][j] + mutation_magnitude
        return ret_matrix

    def __crossover_matrix(self,
                           matrix_a: np.ndarray,
                           matrix_b: np.ndarray,
                           chance_per_genome: float
                           ) -> np.ndarray:
        ret_matrix = np.zeros_like(matrix_a)
        for i in range(matrix_a.shape[0]):
            for j in range(matrix_a.shape[1]):
                genome_chance = np.random.ranf()
                if genome_chance < chance_per_genome:
                    ret_matrix[i][j] = matrix_a[i][j]
                else:
                    ret_matrix[i][j] = matrix_b[i][j]
        return ret_matrix

    def mutate(self, chance_per_genome: float, rate_per_selected: float) -> "TradingIndividual":
        ret_individual = TradingIndividual(self.__input_shape, self.__starting_balance)
        for i in range(len(self.__buy_state_matrices)):
            matrix = self.__buy_state_matrices[i]
            ret_individual.__buy_state_matrices[i] = self.__mutate_matrix(chance_per_genome, rate_per_selected, matrix)
            matrix = self.__sell_state_matrices[i]
            ret_individual.__sell_state_matrices[i] = self.__mutate_matrix(chance_per_genome, rate_per_selected, matrix)
        return ret_individual

    def reset_starting_state(self):
        self.__current_balance = self.__starting_balance
        self.__transaction_state = False
        self.__held_share_price = 0.0
        self.__num_held_shares = 0

    def crossover(self, other: "TradingIndividual", crossover_chance=.5) -> "TradingIndividual":
        ret_individual = TradingIndividual(self.__input_shape, self.__starting_balance)
        for i in range(len(self.__buy_state_matrices)):
            matrix_a = self.__buy_state_matrices[i]
            matrix_b = other.__buy_state_matrices[i]
            ret_individual.__buy_state_matrices[i] = self.__crossover_matrix(matrix_a, matrix_b, crossover_chance)
            matrix_a = self.__sell_state_matrices[i]
            matrix_b = other.__sell_state_matrices[i]
            ret_individual.__sell_state_matrices[i] = self.__crossover_matrix(matrix_a, matrix_b, crossover_chance)
        return ret_individual

    def calculate_fitness(self):
        return self.__current_balance - self.__starting_balance

    def __evaluate_transaction(self,
                               input_data: np.ndarray, state_matrices: List[np.ndarray]
                               ) -> np.ndarray:
        evaluation_ret = input_data
        for i in range(len(state_matrices) - 1):
            evaluation_ret = evaluation_ret @ state_matrices[i]
        return state_matrices[-1] @ evaluation_ret

    def handle_data(self, input_data: np.ndarray, share_price: float):
        if self.__transaction_state:
            evaluation_result = self.__evaluate_transaction(input_data, self.__sell_state_matrices)
            if evaluation_result[0][0] > evaluation_result[0][1] or self.__days_share_held == 5:
                # Indicates we should sell current held shares
                self.__current_balance += self.__num_held_shares * share_price
                self.__num_held_shares = 0
                self.__held_share_price = 0
                self.__transaction_state = False
                self.__days_share_held = 0
            else:
                self.__days_share_held += 1
        else:
            evaluation_result = self.__evaluate_transaction(input_data, self.__buy_state_matrices)
            if evaluation_result[0][0] > evaluation_result[0][1]:
                # Indicates we should buy some shares
                self.__num_held_shares = 100
                self.__current_balance -= self.__num_held_shares * share_price
                self.__held_share_price = share_price
                self.__transaction_state = True

    def predict_data(self, input_data: np.ndarray) -> np.ndarray:
        buy_evaluation_result = self.__evaluate_transaction(input_data, self.__buy_state_matrices)
        sell_evaluation_result = self.__evaluate_transaction(input_data, self.__sell_state_matrices)
        return np.array([
            buy_evaluation_result[0][0] > buy_evaluation_result[0][1],
            sell_evaluation_result[0][0] > sell_evaluation_result[0][1]
        ])

    def load(self, file_handle: IO):
        for i in range(len(self.__buy_state_matrices)):
            self.__buy_state_matrices[i] = pickle.load(file_handle)
        for i in range(len(self.__sell_state_matrices)):
            self.__sell_state_matrices[i] = pickle.load(file_handle)
        self.__starting_balance = pickle.load(file_handle)

    def save(self, file_handle: IO):
        for mat in self.__buy_state_matrices:
            pickle.dump(mat, file_handle)
        for mat in self.__sell_state_matrices:
            pickle.dump(mat, file_handle)
        pickle.dump(self.__starting_balance, file_handle)
