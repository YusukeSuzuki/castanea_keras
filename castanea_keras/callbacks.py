import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback

class TensorBoardImagesExtension(Callback):
    epoch_graph_key = 'TensorBoardImagesExtension_EPOCH_IMAGES'
    batch_graph_key = 'TensorBoardImagesExtension_BATCH_IMAGES'

    def __init__(
            self, tensorboard, images=[],
            max_outputs=3, epoch_freq=1, batch_freq=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard = tensorboard
        self.images = images
        self.max_outputs = max_outputs
        self.epoch_freq = epoch_freq if epoch_freq >= 0 else 0
        self.batch_freq = batch_freq if batch_freq >= 0 else 0

    def set_model(self, model):
        self.sess = K.get_session()
        self.model = model
        self.writer = self.tensorboard.writer
        self.batch_size = self.tensorboard.batch_size

        for image in self.images:
            if type(image) is int:
                tensor = model.get_layer(index=image).output
            elif type(image) is str:
                tensor = model.get_layer(name=image).output
            else:
                tensor = image

            if self.epoch_freq > 0:
                tf.summary.image(
                    tensor.name, tensor, max_outputs=self.max_outputs,
                    collections=[self.epoch_graph_key], family='epoch')
            if self.batch_freq > 0:
                tf.summary.image(
                    tensor.name, tensor, max_outputs=self.max_outputs,
                    collections=[self.batch_graph_key], family='batch')

        self.epoch_merged = tf.summary.merge_all(key=self.epoch_graph_key)
        self.batch_merged = tf.summary.merge_all(key=self.batch_graph_key)

    def _summary_images(self, index, merged, logs=None):
        if not self.validation_data:
            raise ValueError('If printing images, validation_data must be provided.')

        val_data = self.validation_data
        tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)
        if self.model.uses_learning_phase:
            tensors += [K.learning_phase()]

        assert len(val_data) == len(tensors)

        val_size = val_data[0].shape[0]

        assert val_size >= self.batch_size

        if self.model.uses_learning_phase:
            # do not slice the learning phase
            batch_val = [x[:self.batch_size] for x in val_data[:-1]]
            batch_val.append(val_data[-1])
        else:
            batch_val = [x[:self.batch_size] for x in val_data]
        assert len(batch_val) == len(tensors)

        feed_dict = dict(zip(tensors, batch_val))
        result = self.sess.run([merged], feed_dict=feed_dict)
        self.writer.add_summary(result[0], index)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_freq == 0 or epoch % self.epoch_freq != 0:
            return

        self._summary_images(epoch, self.epoch_merged, logs)

    def on_batch_end(self, batch, logs=None):
        if self.batch_freq == 0 or batch % self.batch_freq != 0:
            return

        self._summary_images(batch, self.batch_merged, logs)

