from configparser import ConfigParser, SectionProxy
from general_utils.config import config_util as cfgUtil
from data_providing_module.data_provider_registry import registry, DataConsumerBase
from data_providing_module.data_providers import data_provider_static_names
from stock_data_analysis_module.ml_models.keras_cnn import trainNetwork
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D, AveragePooling2D, Activation
from keras.models import Sequential
from keras.callbacks import History
import keras.optimizers
from typing import Tuple, List
import os
from statistics import mean
import operator


ENABLED_CONFIGURATION_IDENTIFIER = "enabled"


class ScoreIndexPair:

    def __init__(self, score, index):
        self.score = score
        self.index = index


class ModelLayerPair:

    def __init__(self, model: Sequential, layers: List[int]):
        self.model = model
        self.layers = layers


def createModelsFromPreviousLayers(input_shape: Tuple[int], num_categories: int, previous_layers: List[int]) \
        -> List[ModelLayerPair]:
    num_available_layers = sum([len(x[1]) for x in CnnManager.available_layer_choices])
    for i in range(num_available_layers):
        layers = previous_layers.copy()
        layers.append(i)
        ret_model = Sequential()
        ret_model.add(Conv2D(8, (2, 2), input_shape=input_shape, activation='relu'))
        for j in layers:
            seen_args = 0
            for layer, args in CnnManager.available_layer_choices:
                if seen_args + len(args) > j:
                    arg_index = j - seen_args
                    ret_model.add(layer(*args[arg_index]))
                    break
                seen_args += len(args)
        ret_model.add(Flatten())
        ret_model.add(Dense(num_categories, activation='softmax'))
        opt = keras.optimizers.RMSprop()
        ret_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
        yield ModelLayerPair(ret_model, layers)
    pass


def selectTopNModels(model_histories: List[History], n: int = 3) -> List[ScoreIndexPair]:
    model_scores = []
    for i in range(len(model_histories)):
        model_scores.append([0, i])
    total_epochs = len(model_histories[0].history['acc'])
    for epoch in range(total_epochs):
        epoch_scale = (epoch+1) / total_epochs
        accuracies = [x.history['acc'][epoch] for x in model_histories]
        val_accuracies = [x.history['val_acc'][epoch] for x in model_histories]
        avg_val_accuracy = mean(val_accuracies)
        overtrain_scaling = []
        for i in range(len(accuracies)):
            overtrain_scaling.append(val_accuracies[i] / accuracies[i])
            val_accuracies[i] = val_accuracies[i] - avg_val_accuracy
        for i in range(len(accuracies)):
            scaling = epoch_scale * overtrain_scaling[i]
            scaling = 1 / scaling if val_accuracies[i] < 0 else scaling
            model_scores[i][0] += scaling * val_accuracies[i]
    ret_scores = sorted(model_scores, key=operator.itemgetter(0), reverse=True)
    ret_scores = ret_scores[:n]
    for i in range(len(ret_scores)):
        ret_scores[i] = ScoreIndexPair(ret_scores[i][0], ret_scores[i][1])
    return ret_scores


def createModels(input_shape, num_categories, max_added_layers=10,
                 start_layer_range=0) -> List[ModelLayerPair]:
    # the idea here is to start with the base model (2D conv network straight to a flatten, then dense to the number
    # of categories), and return gradually more and more complex networks

    if start_layer_range == 0:
        # Create base model
        ret_model = Sequential()
        ret_model.add(Conv2D(8, (2, 2), input_shape=input_shape, activation='relu'))
        ret_model.add(Flatten())
        ret_model.add(Dense(num_categories, activation='softmax'))
        opt = keras.optimizers.RMSprop()
        ret_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
        yield ModelLayerPair(ret_model, [])
        start_layer_range = 1
    # Create models with more layers
    num_available_layers = sum([len(x[1]) for x in CnnManager.available_layer_choices])
    for i in range(start_layer_range, max_added_layers):
        layer_choices = [0] * i
        while True:
            ret_model = Sequential()
            ret_model.add(Conv2D(8, (2, 2), input_shape=input_shape, activation='relu'))

            # add selected auxiliary layers
            for j in layer_choices:
                seen_args = 0
                for layer, args in CnnManager.available_layer_choices:
                    if seen_args + len(args) > j:
                        arg_index = j - seen_args
                        ret_model.add(layer(*args[arg_index]))
                        break
                    seen_args += len(args)
            ret_model.add(Flatten())
            ret_model.add(Dense(num_categories, activation='softmax'))
            opt = keras.optimizers.RMSprop()
            ret_model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['accuracy'])
            yield ModelLayerPair(ret_model, layer_choices)

            iter_index = None
            for j in range(len(layer_choices)):
                index = -(j+1)
                if not layer_choices[index] == num_available_layers:
                    iter_index = index
                    break
            if iter_index is None:
                break
            for j in range(iter_index+1, len(layer_choices)):
                x = layer_choices[j]
                layer_choices[j] = x if not x == num_available_layers else 0
            layer_choices[iter_index] += 1


def writeHistory(model_identifier, history: History, out_obj):
    print(model_identifier, file=out_obj)
    for i in range(len(history.history['acc'])):
        print('\t%d:' % i, history.history['acc'][i], history.history['val_acc'][i], file=out_obj)


def createModelsTrialByFire(data, passback, output_dir, max_layers_to_add: int = 8):
    x_train, y_train, x_test, y_test = data
    x_train = x_train.astype('float')
    y_train = y_train.astype('float')
    y_test = y_test.astype('float')
    x_test = x_test.astype('float')
    data_shape = x_train[0].shape
    cats = len(y_train[0])
    models = createModelsFromPreviousLayers(data_shape, num_categories=cats, previous_layers=[])
    generators = [models]
    for i in range(max_layers_to_add):
        iter_best_models = []
        for generator in generators:
            histories = []
            for model_layers in generator:
                hist = trainNetwork(x_train, y_train, model_layers.model, epochs=50, validation_data=(x_test, y_test))
                histories.append([hist, model_layers.layers])
                pass
            best_models = selectTopNModels([x[0] for x in histories])
            if len(generators) == 1:
                iter_best_models = [histories[x.index][1] for x in best_models]
            else:
                iter_best_models.append(histories[best_models[0].index][1])
            for best_candidate in best_models:
                history, layers = histories[best_candidate.index]
                model_identifer = passback + '-'.join([str(x) for x in layers])
                with open(output_dir + os.sep + model_identifer + '.mlinf', 'w') as fileHandle:
                    writeHistory(model_identifer, history, fileHandle)
        generators = []
        for best_layers in iter_best_models:
            generators.append(createModelsFromPreviousLayers(
                data_shape, num_categories=cats, previous_layers=best_layers))


class CnnManager (DataConsumerBase):

    def predictData(self, data, passback, in_model_dir):
        pass

    def load_configuration(self, parser: "ConfigParser"):
        section = cfgUtil.create_type_section(parser, self)
        if not parser.has_option(section.name, ENABLED_CONFIGURATION_IDENTIFIER):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, ENABLED_CONFIGURATION_IDENTIFIER)
        if not enabled:
            registry.deregisterConsumer(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID, self)
            registry.deregisterConsumer(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID, self)

    def write_default_configuration(self, section: "SectionProxy"):
        section[ENABLED_CONFIGURATION_IDENTIFIER] = 'False'

    available_layer_choices = [(Conv2D, [(8, (2, 2)), (16, (2, 2)), (32, (2, 2)), (64, (2, 2))]),
                               (Dense, [[8], [16], [32], [64]]), (Dropout, [[.1], [.15], [.2]]),
                               (MaxPool2D, [[2, 2], [3, 3]]),
                               (AveragePooling2D, [[2, 2], [3, 3]]), (Activation, [['relu']])]

    def __init__(self):
        super(CnnManager, self).__init__()
        registry.registerConsumer(data_provider_static_names.CLUSTERED_BLOCK_PROVIDER_ID, self,
                                  [['hist_date', 'adj_close', 'opening_price', 'volume_data', 'high_price'],
                                   [1]], passback='CombinedDataCNN')
        registry.registerConsumer(data_provider_static_names.SPLIT_BLOCK_PROVIDER_ID, self,
                                  [['hist_date', 'adj_close', 'opening_price', 'volume_data', 'high_price'],
                                   [1]], passback='SplitDataCNN')

    def consumeData(self, data, passback, output_dir):
        if not type(data) == dict:
            return
            createModelsTrialByFire(data, passback, output_dir)
        else:
            for ticker, model_data in data.items():
                createModelsTrialByFire(model_data, passback + '_' + ticker, output_dir)


consumer = CnnManager()
