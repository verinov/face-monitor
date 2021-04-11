import tensorflow as tf


class PerceptionLoss(tf.keras.losses.Loss):
    def __init__(self, input_shape):
        super().__init__()
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            #     pooling='avg',
        )
        self.feat_extraction_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer(name="block1a_activation").output,
        )
        self.feat_extraction_model.trainable = False

    def call(self, y_true, y_pred):
        y_true_features = self.feat_extraction_model(y_true)
        y_pred_features = self.feat_extraction_model(y_pred)
        x = tf.reduce_sum(tf.square(y_true_features - y_pred_features), axis=(1, 2, 3))
        return tf.reduce_mean(x)