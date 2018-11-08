import sys
from traceback import format_exception

import tensorflow as tf
from .datareader import DataReader

def read_op(csv_records, size, num_categories, shuffle=True):
    try:
        queue = tf.train.string_input_producer(
            csv_records, shuffle=shuffle)
        reader = tf.IdentityReader()
        _, record = reader.read(queue)
        path, label = tf.decode_csv(record, [[''],[0]])
        image_raw = tf.read_file(path)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        image = tf.image.resize_images(image, size)
        label = tf.one_hot(label, num_categories)
        x = image
        y = label
    except:
        print(format_exception(*sys.exc_info()))

    return x, y

def categorical_image_generator(
    batch_size, list_of_categorical_imagepath, size=(128,128), shuffle=True):
    csv_records= []
    num_categories = len(list_of_categorical_imagepath)
    '''
    Parameters
    ----------
    batch_size : int
        size of minibatch
    list_of_categorical_imagepath : list
        list of image file path like [[cat0image0, cat0image1,...],[cat1image0,...],[...]]
    size : tuple
        size of image (heigh, width)
    shuffle : bool
        do shuffling or not

    Returns
    -------
    iterable image generator : generator
    '''

    for i,l in enumerate(list_of_categorical_imagepath):
        for p in l:
            csv_records.append('"{}",{}'.format(p,i))

    return DataReader(batch_size, read_op,
        args=[csv_records, size, num_categories],
        kwargs=dict(shuffle=shuffle))

