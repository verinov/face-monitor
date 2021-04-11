from math import prod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.utils.conv_utils import conv_output_length


class SplitLayer(keras.layers.Layer):
    def __init__(self, num_splits=2, axis=1, clip_range=None):
        super().__init__()
        self.num_splits = num_splits
        self.axis = axis
        self.clip_range = clip_range

    def call(self, inputs):
        mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
        if self.clip_range is not None:
            # soft clip logvar
            min_value, max_value = self.clip_range
            logvar = tf.sigmoid(logvar) * (max_value - min_value) + min_value
        return mean, logvar


class ReparameterizeLayer(keras.layers.Layer):
    def call(self, mean, logvar):
        x = tf.random.normal(shape=tf.shape(mean))
        return mean + x * tf.exp(logvar * 0.5)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    return tf.reduce_sum(
        -0.5
        * (
            (sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + tf.math.log(2.0 * np.pi)
        ),
        axis=raxis,
    )


def standard_normal_kl(mean, logvar, raxis=1):
    # KL[N(mean, var) || N(0, 1)]
    return 0.5 * tf.reduce_sum(tf.exp(logvar) + tf.square(mean) - 1.0 - logvar, axis=1)


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(
        self, input_shape, decoder_input_shape, latent_dim, encoder_base, decoder_base
    ):
        """
        input_shape: Tuple, (width, height, channels)
        latent_dim: int
        """
        super().__init__()
        self.latent_dim = latent_dim

        x = encoder_input = keras.Input(shape=input_shape, name="image")
        x = encoder_base(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dense(latent_dim + latent_dim)(x)
        encoder_mean, encoder_logvar = SplitLayer(clip_range=(-2, 2))(x)
        self.encoder = keras.Model(
            encoder_input, (encoder_mean, encoder_logvar), name="encoder"
        )
        self.encoder.summary()

        self.reparameterize_layer = ReparameterizeLayer()

        x = decoder_input = keras.Input(shape=latent_dim, name="embedding")
        x = keras.layers.Dense(
            units=prod(decoder_input_shape),
            # activation="relu",
        )(x)
        x = keras.layers.Reshape(target_shape=decoder_input_shape)(x)
        x = decoder_base(x)

        decoder_output = x

        self.decoder = keras.Model(decoder_input, decoder_output, name="decoder")
        self.decoder.summary()

    def call(self, x, training=None):
        encoder_mean, encoder_logvar = self.encoder(x, training=training)
        kl = tf.reduce_mean(standard_normal_kl(encoder_mean, encoder_logvar))
        self.add_metric(kl, name="EncodingKL")
        self.add_loss(kl)

        if training:
            encoder_sample = self.reparameterize_layer(encoder_mean, encoder_logvar)
            # KL[Q(z|X) || P(z)]
            # NB: negative E[log P(X|z)] must be added to the loss elsewhere
            return self.decoder(encoder_sample, training=training)
        else:
            return self.decoder(encoder_mean, training=training)

    # def fit():
    # def evaluate():
