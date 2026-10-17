"""PlantVillage augmentation helpers. Apply augment_image to training images only."""
import math
import tensorflow as tf

IMG_SIZE = 224
MAX_CUTOUT_AREA = 0.20

def load_rgb_image(path, img_size=IMG_SIZE):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_size, img_size], antialias=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape([img_size, img_size, 3])
    return image


def _rotate_image(image, angle_degrees):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    angle = tf.cast(angle_degrees, tf.float32) * (math.pi / 180.0)
    cosine, sine = tf.cos(angle), tf.sin(angle)
    cx = (tf.cast(width, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(height, tf.float32) - 1.0) / 2.0
    transform = tf.stack([
        cosine, -sine, cx - cosine * cx + sine * cy,
        sine, cosine, cy - sine * cx - cosine * cy,
        0.0, 0.0,
    ])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=tf.reshape(transform, [1, 8]),
        output_shape=tf.stack([height, width]),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]


def _gaussian_blur(image, sigma):
    coords = tf.range(-2, 3, dtype=tf.float32)
    kernel_1d = tf.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d /= tf.reduce_sum(kernel_1d)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = tf.tile(kernel_2d[:, :, None, None], [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(image[None, ...], kernel, [1, 1, 1, 1], "SAME")[0]


def _random_cutout_mask(height, width, max_area=MAX_CUTOUT_AREA):
    area = tf.random.uniform([], 0.02, max_area) * tf.cast(height * width, tf.float32)
    aspect = tf.random.uniform([], 0.5, 2.0)
    cut_h = tf.clip_by_value(tf.cast(tf.floor(tf.sqrt(area / aspect)), tf.int32), 1, height)
    cut_w = tf.clip_by_value(tf.cast(tf.floor(tf.sqrt(area * aspect)), tf.int32), 1, width)
    y0 = tf.random.uniform([], 0, height - cut_h + 1, dtype=tf.int32)
    x0 = tf.random.uniform([], 0, width - cut_w + 1, dtype=tf.int32)
    ys, xs = tf.range(height)[:, None], tf.range(width)[None, :]
    return (ys >= y0) & (ys < y0 + cut_h) & (xs >= x0) & (xs < x0 + cut_w)


def _apply_cutout(image, max_area=MAX_CUTOUT_AREA):
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    mask = _random_cutout_mask(height, width, max_area)[:, :, None]
    fill = tf.broadcast_to(tf.reduce_mean(image, axis=[0, 1], keepdims=True), tf.shape(image))
    return tf.where(mask, fill, image)


def _maybe(probability, transform, image):
    return tf.cond(tf.random.uniform([]) < probability, lambda: transform(image), lambda: image)


def augment_image(image, strong=False):
    """Applying random augmentation to one RGB float image in [0,1]. Used for training part only."""
    image_shape = image.shape
    image = tf.cast(image, tf.float32)
    strong = tf.convert_to_tensor(strong, dtype=tf.bool)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    max_rotation = tf.where(strong, 40.0, 30.0)
    angle = tf.random.uniform([], -max_rotation, max_rotation)
    image = _rotate_image(image, angle)

    image *= tf.random.uniform([], 0.60, 1.40)
    image = tf.image.adjust_contrast(
        tf.clip_by_value(image, 0.0, 1.0),
        tf.random.uniform([], tf.where(strong, 0.75, 0.80), tf.where(strong, 1.25, 1.20)),
    )
    image = tf.image.adjust_saturation(
        tf.clip_by_value(image, 0.0, 1.0),
        tf.random.uniform([], tf.where(strong, 0.75, 0.80), tf.where(strong, 1.25, 1.20)),
    )
    hue_limit = tf.where(strong, 0.03, 0.02)
    image = tf.image.adjust_hue(
        tf.clip_by_value(image, 0.0, 1.0), tf.random.uniform([], -hue_limit, hue_limit)
    )

    image = _maybe(
        tf.where(strong, 0.30, 0.20),
        lambda x: _gaussian_blur(x, tf.random.uniform([], 0.5, 1.2)), image,
    )
    noise_std = tf.where(strong, 0.03, 0.02)
    image = _maybe(
        tf.where(strong, 0.30, 0.20),
        lambda x: x + tf.random.normal(tf.shape(x), stddev=noise_std), image,
    )
    image = _maybe(
        tf.where(strong, 0.35, 0.25),
        lambda x: _apply_cutout(x), image,
    )

    image = tf.clip_by_value(image, 0.0, 1.0)
    image.set_shape(image_shape)
    return image
