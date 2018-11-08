import tensorflow as tf
import multiprocessing as mp

class DataReader():
    def __init__(self, batch_size, op, args=[], kwargs={}, queue_max=5000, queue_min=1000):
        self.graph = tf.Graph()
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config, graph=self.graph)
        self.coordinator=tf.train.Coordinator()
        self.threads = None

        with self.graph.as_default(), tf.device('/cpu:0'):
            self.batch = tf.train.shuffle_batch(
                [*op(*args, **kwargs)], batch_size, queue_max, queue_min,
                num_threads=mp.cpu_count()*4)

        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coordinator)

    def _pumpup(self):
        pass

    def __iter__(self):
        return self

    def __del__(self):
        self.coordinator.request_stop()

        if self.threads:
            self.coordinator.join(self.threads)

    def __next__(self):
        return tuple(self.sess.run(self.batch))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

