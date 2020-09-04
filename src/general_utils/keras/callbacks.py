"""Module for storage of callback classes for Keras Model training

"""
import keras

from general_utils.logging import logger


class HighestAccuracyModelStorageCallback(keras.callbacks.Callback):

    def __init__(self, model: keras.Model, file_path: str, do_logging: bool = False):
        super(HighestAccuracyModelStorageCallback, self).__init__()
        self._model_ref = model
        self._out_file = file_path
        self._highest_val_accuracy = 0
        self._do_log = do_logging

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            accuracy_name = 'val_acc'
            if 'val_acc' not in logs:
                if 'val_accuracy' in logs:
                    accuracy_name = 'val_accuracy'
                else:
                    return
            if logs[accuracy_name] >= self._highest_val_accuracy:
                self._highest_val_accuracy = logs[accuracy_name]
                self._model_ref.save(self._out_file)
                if self._do_log:
                    logger.logger.log(logger.INFORMATION,
                                      f"Saving model to {self._out_file} on epoch {epoch}. "
                                      f"Current accuracy: {self._highest_val_accuracy}")
