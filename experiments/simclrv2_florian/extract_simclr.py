"""
Adapted from https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb

Unmodified from Florian's codebase
    https://github.com/ftramer/Handcrafted-DP/blob/main/transfer/extract_simclr.py

Run
    python -m experiments.simclrv2_florian.extract_simclr --dataset "cifar-10" &
    python -m experiments.simclrv2_florian.extract_simclr --dataset "cifar-10.2" &
    python -m experiments.simclrv2_florian.extract_simclr --dataset "cinic-10" &
    python -m experiments.simclrv2_florian.extract_simclr --dataset "cinic-10-pure" &
"""

import numpy as np
import tensorflow.compat.v1 as tf
import tqdm

from .download import available_simclr_models

# Tensorflow is just annoying...
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.disable_eager_execution()
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
import os
import fire

CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
                tf.cast(p, tf.float32)),
        lambda: func(x),
        lambda: x)


def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
    """Compute aspect ratio-preserving shape for central crop.
    The resulting shape retains `crop_proportion` along one side and a proportion
    less than or equal to `crop_proportion` along the other side.
    Args:
      image_height: Height of image to be cropped.
      image_width: Width of image to be cropped.
      aspect_ratio: Desired aspect ratio (width / height) of output.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      crop_height: Height of image after cropping.
      crop_width: Width of image after cropping.
    """
    image_width_float = tf.cast(image_width, tf.float32)
    image_height_float = tf.cast(image_height, tf.float32)

    def _requested_aspect_ratio_wider_than_image():
        crop_height = tf.cast(tf.rint(
            crop_proportion / aspect_ratio * image_width_float), tf.int32)
        crop_width = tf.cast(tf.rint(
            crop_proportion * image_width_float), tf.int32)
        return crop_height, crop_width

    def _image_wider_than_requested_aspect_ratio():
        crop_height = tf.cast(
            tf.rint(crop_proportion * image_height_float), tf.int32)
        crop_width = tf.cast(tf.rint(
            crop_proportion * aspect_ratio *
            image_height_float), tf.int32)
        return crop_height, crop_width

    return tf.cond(
        aspect_ratio > image_width_float / image_height_float,
        _requested_aspect_ratio_wider_than_image,
        _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
    """Crops to center of image and rescales to desired size.
    Args:
      image: Image Tensor to crop.
      height: Height of image to be cropped.
      width: Width of image to be cropped.
      crop_proportion: Proportion of image to retain along the less-cropped side.
    Returns:
      A `height` x `width` x channels Tensor holding a central crop of `image`.
    """
    shape = tf.shape(image)
    image_height = shape[0]
    image_width = shape[1]
    crop_height, crop_width = _compute_crop_shape(
        image_height, image_width, height / width, crop_proportion)
    offset_height = ((image_height - crop_height) + 1) // 2
    offset_width = ((image_width - crop_width) + 1) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    image = tf.image.resize_bicubic([image], [height, width])[0]

    return image


def preprocess_for_eval(image, height, width, crop=True):
    """Preprocesses the given image for evaluation.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      crop: Whether or not to (center) crop the test images.
    Returns:
      A preprocessed image `Tensor`.
    """
    if crop:
        image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image


def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
    """Preprocesses the given image.
    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.
      is_training: `bool` for whether the preprocessing is for training.
      color_distort: whether to apply the color distortion.
      test_crop: whether or not to extract a central crop of the images
          (as for standard ImageNet evaluation) during the evaluation.
    Returns:
      A preprocessed image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # if is_training:
    #  return preprocess_for_train(image, height, width, color_distort)
    # else:
    return preprocess_for_eval(image, height, width, test_crop)


def _preprocess(x):
    x = preprocess_image(x, 224, 224, is_training=False, color_distort=False)
    return x


def _extract_single(
    model_name="r50_2x_sk1",
    evaluate=False,
    dataset="cifar-10",  # One of cifar-10, cinic-10, cifar-10.2, cinic-10-pure.
    batch_size=200,
):
    if dataset == "cifar-10":
        (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
        xtrain = xtrain.astype(np.float32) / 255.0
        xtest = xtest.astype(np.float32) / 255.0
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
    elif dataset in ("cinic-10", "cinic-10-pure"):
        import imageio
        from swissknife import utils

        img_shape = (32, 32, 3)  # Prescribed shape -- some images don't have this shape => fail!
        base_dir = "/nlp/scr/lxuechen/data/CINIC-10"

        def get_label(path):
            class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for class_label_index, class_label in enumerate(class_labels):
                if class_label in path:
                    return class_label_index
            raise ValueError

        def get_images_and_labels(path):
            images = []
            labels = []
            for i, img_path in tqdm.tqdm(enumerate(utils.listfiles(path)), desc=path):
                if dataset == "cinic-10-pure":  # Exclude the original CIFAR-10.
                    if 'cifar10-train' in img_path or 'cifar10-test' in img_path:
                        continue
                img = imageio.imread(img_path)
                if img.shape != img_shape:
                    continue
                images.append(img)
                labels.append(get_label(img_path))
            images = np.stack(images, axis=0)
            labels = np.array(labels)

            images = images.astype(np.float32) / 255.0
            labels = labels.reshape(-1)
            return images, labels

        # train
        train_path = utils.join(base_dir, 'train')
        train_npz_path = utils.join(base_dir, f'{dataset}_train.npz')
        if not os.path.exists(train_npz_path):
            xtrain, ytrain = get_images_and_labels(path=train_path)
            np.savez(train_npz_path, images=xtrain, labels=ytrain)
        else:
            train_file = np.load(train_npz_path)
            xtrain, ytrain = train_file["images"], train_file["labels"]

        # test
        test_path = utils.join(base_dir, 'test')
        test_npz_path = utils.join(base_dir, f'{dataset}_test.npz')
        if not os.path.exists(test_npz_path):
            xtest, ytest = get_images_and_labels(path=test_path)
            np.savez(test_npz_path, images=xtest, labels=ytest)
        else:
            test_file = np.load(test_npz_path)
            xtest, ytest = test_file["images"], test_file["labels"]

    elif dataset == "cifar-10.2":
        train_file = np.load(
            '/home/lxuechen_stanford_edu/software/swissknife/experiments/priv_fair/data/cifar-10.2-master'
            '/cifar102_train.npz',
        )
        test_file = np.load(
            '/home/lxuechen_stanford_edu/software/swissknife/experiments/priv_fair/data/cifar-10.2-master'
            '/cifar102_test.npz',
        )
        xtrain = train_file["images"]
        xtest = test_file["images"]

        ytrain = train_file["labels"]
        ytest = test_file["labels"]

        xtrain = xtrain.astype(np.float32) / 255.0
        xtest = xtest.astype(np.float32) / 255.0

        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    x = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)
    x_preproc = tf.map_fn(_preprocess, x)
    print(x_preproc.get_shape().as_list())

    hub_path = f'gs://simclr-checkpoints/simclrv2/pretrained/{model_name}/hub/'
    module = hub.Module(hub_path, trainable=False)
    features = module(inputs=x_preproc, signature='default')
    print(features.get_shape().as_list())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("model loaded!")

    features_train = []
    for i in tqdm.tqdm(range(len(xtrain) // batch_size), desc="train batches"):
        x_batch = xtrain[i * batch_size:(i + 1) * batch_size]
        f = sess.run(features, feed_dict={x: x_batch})
        features_train.append(f)
    if batch_size * (len(xtrain) // batch_size) < len(xtrain):
        x_batch = xtrain[(i + 1) * batch_size:]
        f = sess.run(features, feed_dict={x: x_batch})
        features_train.append(f)

    features_train = np.concatenate(features_train, axis=0)
    print(features_train.shape)
    assert features_train.shape[0] == len(xtrain)

    features_test = []
    for i in tqdm.tqdm(range(len(xtest) // batch_size), desc="test batches"):
        x_batch = xtest[i * batch_size:(i + 1) * batch_size]
        f = sess.run(features, feed_dict={x: x_batch})
        features_test.append(f)
    if batch_size * (len(xtest) // batch_size) < len(xtest):
        x_batch = xtest[(i + 1) * batch_size:]
        f = sess.run(features, feed_dict={x: x_batch})
        features_test.append(f)

    features_test = np.concatenate(features_test, axis=0)
    print(features_test.shape)
    assert features_test.shape[0] == len(xtest)

    base_dir = f"/nlp/scr/lxuechen/features/{dataset}"
    os.makedirs(base_dir, exist_ok=True)
    np.savez(f"{base_dir}/simclr_{model_name}_train.npz", features=features_train, labels=ytrain)
    np.savez(f"{base_dir}/simclr_{model_name}_test.npz", features=features_test, labels=ytest)

    if evaluate:
        mean = np.mean(features_train, axis=0)
        var = np.var(features_train, axis=0)

        features_train_norm = (features_train - mean) / np.sqrt(var + 1e-5)
        features_test_norm = (features_test - mean) / np.sqrt(var + 1e-5)

        for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train, ytrain)
            print(C, clf.score(features_train, ytrain), clf.score(features_test, ytest))

            clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm, ytrain)
            print(C, clf.score(features_train_norm, ytrain), clf.score(features_test_norm, ytest))


def main(**kwargs):
    for model_name in available_simclr_models:
        _extract_single(model_name=model_name, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
