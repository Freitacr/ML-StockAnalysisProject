"""Module for storage of callback classes for Keras Model training

"""
import keras

from general_utils.logging import logger


class HighestAccuracyModelStorageCallback(keras.callbacks.Callback):

    def __init__(self, model: keras.Model, file_path: str):
        super(HighestAccuracyModelStorageCallback, self).__init__()
        self._model_ref = model
        self._out_file = file_path
        self._highest_val_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            if 'val_acc' in logs:
                if logs['val_acc'] >= self._highest_val_accuracy:
                    self._highest_val_accuracy = logs['val_acc']
                    self._model_ref.save(self._out_file)
                    logger.logger.log(logger.INFORMATION,
                                      f"Saving model to {self._out_file} on epoch {epoch}. "
                                      f"Current accuracy: {self._highest_val_accuracy}")
