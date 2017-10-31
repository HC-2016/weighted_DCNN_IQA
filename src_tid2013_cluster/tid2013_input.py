"""
@File         : tid2013_input.py
@Time         : 2017/6/22 
@Author       : Chen Huang
@Update       : 
@Discription  : convert original data to standard tensorflow data '.tfrecords';
                generate image and label batch.
"""


from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import gzip
import numpy
import PIL.Image as Image

from tensorflow.python.platform import gfile
import tensorflow as tf

# dirs
DATA_DIR = '../data'
LOG_DIR = '../logs'

# path of mnist
DIR = '/home/Eric/data'
DATABASE = 'tid2013'
MOS_PATH = DIR + '/' + DATABASE + '/mos_with_names.txt'
REF_PATH = DIR + '/' + DATABASE + '/reference_images/'
DIS_PATH = DIR + '/' + DATABASE + '/distorted_images/'

# information of mnist
HEIGHT = 384
WIDTH = 512
DEPTH = 3

PATCH_SIZE = 32
NUM_PATCHES_PER_IMAGE = 32

TRAIN_DATA_NUM = 15
VAL_DATA_NUM = 5
TEST_DATA_NUM = 5
NUM_PER_IMAGE = 5*24

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        :param f: file object. a file object that can be passed into a gzip reader.

    Returns:
        :return data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
        :exception ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)

        return data


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        :param f: file object. A file object that can be passed into a gzip reader.
        :param one_hot: bool. Does one hot encoding for the result.
        :param num_classes: int. Number of classes for the one hot encoding.

    Returns:
        :returns labels: ndarray. a 1D uint8 numpy array.

    Raises:
        :exception ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)

        return labels


def load_img(path, gray_scale=False):
    """ Load image and convert to Image object.

    Args:
        :param path:str. the path of the image file.
        :param gray_scale:bool. gray or color.

    Return:
        :return img:Image object. an instance of Image object.
    """
    img = Image.open(path)
    if gray_scale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(x, y, z, filename):
    """Converts data to tfrecords.

    Args:
      :param x, y: list - [img1, img2, ...].
                    img: ndarray.
      :param name: str. 
    """
    if not gfile.Exists(filename):
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(NUM_PER_IMAGE):
            image_x = x[index].tostring()
            image_y = y[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _float_feature(z[index]),
                'image_x': _bytes_feature(image_x),
                'image_y': _bytes_feature(image_y)
            }))
            writer.write(example.SerializeToString())
        writer.close()


def load_data():
    """Load database, convert to Examples and write the result to TFRecords."""
    # Load data
    text_file = open(MOS_PATH, "r")
    lines = text_file.readlines()
    text_file.close()

    ref_img_set = []
    dis_img_set = []
    mos_set = []
    for line in lines:
        mos, name = line.rstrip().split(' ', 2)  # (5.51, i01_01_1.bmp)
        ref_name = name.split('_')[0] + '.bmp'  # i01.bmp

        path = REF_PATH + ref_name.lower()
        ref_img_set.append(numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8))  # ndarray(0,255)
        path = DIS_PATH + name.lower()
        dis_img_set.append(numpy.asarray(load_img(path, gray_scale=False), dtype=numpy.uint8))
        mos_set.append(float(mos))

    # Convert to Examples and write the result to TFRecords.
    for i in range(TRAIN_DATA_NUM+VAL_DATA_NUM+TEST_DATA_NUM):
        x = [ref_img_set[j] for j in numpy.arange(i * NUM_PER_IMAGE, (i + 1) * NUM_PER_IMAGE)]
        y = [dis_img_set[j] for j in numpy.arange(i * NUM_PER_IMAGE, (i + 1) * NUM_PER_IMAGE)]
        z = [mos_set[j] for j in numpy.arange(i * NUM_PER_IMAGE, (i + 1) * NUM_PER_IMAGE)]

        # import matplotlib.pyplot as plt
        # plt.figure('subplot')
        # plt.subplot(2, 2, 1)
        # plt.imshow(ref_img_set[0] / 255.0)
        # plt.subplot(2, 2, 2)
        # plt.imshow(dis_img_set[0] / 255.0)
        # plt.subplot(2, 2, 3)
        # plt.imshow(ref_img_set[4] / 255.0)
        # plt.subplot(2, 2, 4)
        # plt.imshow(dis_img_set[4] / 255.0)
        # plt.show()

        if not gfile.Exists(os.path.join(DATA_DIR)):
            os.makedirs(os.path.join(DATA_DIR))

        filename = os.path.join(DATA_DIR, 'image_' + str(i) + '.tfrecords')
        convert_to(x, y, z, filename)


def read_and_decode(filename_queue):
    """Reads and parses examples from data files .tfrecords.

    Args:
        :param filename_queue: queue. A queue of strings with the filenames to read from. 

    Returns:
        :return result: DataRecord. An object representing a single example, with the following fields:
            label: an int32 Tensor.
            image_x, image_y: a [height*width*depth] uint8 Tensor with the image data.
    """

    class DataRecord(object):
        pass

    result = DataRecord()
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature([], tf.float32),
            'image_x': tf.FixedLenFeature([], tf.string),
            'image_y': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor to a uint8 tensor
    result.image_x = tf.decode_raw(features['image_x'], tf.uint8)
    result.image_y = tf.decode_raw(features['image_y'], tf.uint8)
    result.image_x.set_shape([HEIGHT*WIDTH*DEPTH])
    result.image_y.set_shape([HEIGHT*WIDTH*DEPTH])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    result.label = features['label']

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
        :param image: tuple - (image_x, image_y). 
                imagex, image_y: 3-D Tensor of [height, width, 3] of type.float32.
        :param label: 1-D Tensor of type.float32
        :param min_queue_examples: int32, minimum number of samples to retain 
        in the queue that provides of batches of examples.
        :param batch_size: Number of images per batch.
        :param shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
        :return images_x, images_y: Images. 4D tensor of [batch_size, height, width, 3] size.
        :return labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 8
    if shuffle:
        images_x, images_y, label_batch = tf.train.shuffle_batch(
            [image[0], image[1], label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images_x, images_y, label_batch = tf.train.batch(
            [image[0], image[1], label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    #tf.summary.image('batch_images_x', tensor=images_x, max_outputs=4)
    #tf.summary.image('batch_images_y', tensor=images_y, max_outputs=4)

    return images_x, images_y, label_batch


def random_sample(images_x, images_y, patch_size, num_patches):
    """random sample patch pairs from image pairs.
    
    Args:
        :param images_x, images_y: tensor - (batch_size, height, width, depth). 
        :param patch_size: int. 
        :param num_patches: int. we crop num_patches patches from each image pair.
        
    Returns:
        :return patches_x, patches_y: tensor - (batch_size*num_patches, patch_size, patch_size, depth). 
    """
    patches_x = []
    patches_y = []

    images_xy = tf.concat([images_x, images_y], axis=3)
    for i in range(images_x.get_shape()[0].value):
        for j in range(num_patches):
            # Randomly crop a [height, width] section of the image.
            patch_xy = tf.random_crop(images_xy[i, :, :, :], [patch_size, patch_size, DEPTH*2])

            patches_x.append(patch_xy[:, :, :3])
            patches_y.append(patch_xy[:, :, 3:])

    patches_x = tf.convert_to_tensor(value=patches_x, dtype=tf.float32, name='sampled_patches_x')
    patches_y = tf.convert_to_tensor(value=patches_y, dtype=tf.float32, name='sampled_patches_y')

    return patches_x, patches_y


def distorted_inputs(filenames, batch_size):
    """Construct distorted input for training using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (patches_x, patches_y, labels).
                patches_x, patches_y: tensors - (batch_size*num_patches, patch_size, patch_size, depth). 
                lables: tensors - (batch_size).
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.variable_scope('input'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename queue.
        result = read_and_decode(filename_queue)

        # OPTIONAL: Could reshape into a image and apply distortionshere.
        reshaped_image_x = tf.reshape(result.image_x, [HEIGHT, WIDTH, DEPTH])
        reshaped_image_y = tf.reshape(result.image_y, [HEIGHT, WIDTH, DEPTH])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        distorted_image_x = tf.cast(reshaped_image_x, tf.float32) * (1. / 255) - 0.5
        distorted_image_y = tf.cast(reshaped_image_y, tf.float32) * (1. / 255) - 0.5

        # # Randomly flip the image horizontally.
        # distorted_image = tf.image.random_flip_left_right(distorted_image)
        #
        # # Because these operations are not commutative, consider randomizing
        # # the order their operation.
        # distorted_image = tf.image.random_brightness(distorted_image,
        #                                              max_delta=63)
        # distorted_image = tf.image.random_contrast(distorted_image,
        #                                            lower=0.2, upper=1.8)
        #
        # # Subtract off the mean and divide by the variance of the pixels.
        # distorted_image = tf.image.per_image_standardization(distorted_image)
        #
        # # Set the shapes of tensors.
        # distorted_image.set_shape([patch_size, patch_size, result.depth])
        # result.label.set_shape([1])
        label = result.label

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 1000
        # print('Filling queue with %d mnist images before starting to train or validation. '
        #       'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        images_x, images_y, labels = \
            _generate_image_and_label_batch(image=(distorted_image_x, distorted_image_y), label=label,
                                               min_queue_examples=min_queue_examples,
                                               batch_size=batch_size,
                                               shuffle=True)

        # Random crop patches from images
        patches_x, patches_y = random_sample(images_x, images_y, PATCH_SIZE, NUM_PATCHES_PER_IMAGE)

        # Display the training images in the visualizer.
        #tf.summary.image('patches_x', tensor=patches_x, max_outputs=4)
        #tf.summary.image('patches_y', tensor=patches_y, max_outputs=4)

        return patches_x, patches_y, labels


def inputs(filenames, batch_size):
    """Construct input without distortion for MNIST using the Reader ops.

    Args:
        :param filenames: list - [str1, str2, ...].
        :param batch_size: int. 

    Returns:
       :returns: tuple - (images, labels).
                images: tensors - [batch_size, height*width*depth].
                lables: tensors - [batch_size].
    """
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.variable_scope('input_evaluation'):
        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(string_tensor=filenames)

        # Even when reading in multiple threads, share the filename
        # queue.
        result = read_and_decode(filename_queue)

        # OPTIONAL: Could reshape into a image and apply distortionshere.
        reshaped_image_x = tf.reshape(result.image_x, [HEIGHT, WIDTH, DEPTH])
        reshaped_image_y = tf.reshape(result.image_y, [HEIGHT, WIDTH, DEPTH])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image_x = tf.cast(reshaped_image_x, tf.float32) * (1. / 255) - 0.5
        image_y = tf.cast(reshaped_image_y, tf.float32) * (1. / 255) - 0.5
        label = result.label

        # Ensure that the random shuffling has good mixing properties.
        min_queue_examples = 500
        # print('Filling queue with %d mnist images before starting to train or validation. '
        #       'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        images_x, images_y, labels = \
            _generate_image_and_label_batch(image=(image_x, image_y), label=label,
                                               min_queue_examples=min_queue_examples,
                                               batch_size=batch_size,
                                               shuffle=True)

        # Random crop patches from images
        patches_x, patches_y = random_sample(images_x, images_y, PATCH_SIZE, NUM_PATCHES_PER_IMAGE)

        return patches_x, patches_y, labels