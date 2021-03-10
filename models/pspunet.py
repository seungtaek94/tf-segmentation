import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, \
    BatchNormalization, Activation, AveragePooling2D, ReLU, Dropout

class global_average_pooling(tf.keras.Model):
    def __init__(self, _input_shape, gap_size):
        super(global_average_pooling, self).__init__()
        self.w, self.h, self.c = (_input_shape[1:])
        self.avg_pool = AveragePooling2D((self.w / gap_size, self.h / gap_size))
        self.conv = Conv2D(self.c // 4, (1, 1), padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = tf.raw_ops.ResizeBilinear(images=x, size=(self.w, self.h))

        return x


def contract_path(input_shape):
    filter_size = 32
    input = tf.keras.layers.Input(shape=input_shape)
    x = Conv2D(filter_size, (3, 3), padding="same", activation="relu")(input)
    x = Conv2D(filter_size, (3, 3), padding="same", activation="relu", name="copy_crop1")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filter_size * 2, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filter_size * 2, (3, 3), padding="same", activation="relu", name="copy_crop2")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filter_size * 4, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filter_size * 4, (3, 3), padding="same", activation="relu", name="copy_crop3")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filter_size * 8, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filter_size * 8, (3, 3), padding="same", activation="relu", name="copy_crop4")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filter_size * 16, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(filter_size * 16, (3, 3), padding="same", activation="relu", name="last_layer")(x)
    x = Dropout(0.5)(x)
    contract_path = tf.keras.Model(inputs=input, outputs=x)
    return contract_path


def pspunet(input_shape, n_classes):
    filter_size = 32
    contract_model = contract_path(input_shape=input_shape)
    layer_names = ["copy_crop1", "copy_crop2", "copy_crop3", "copy_crop4"]
    layers = [contract_model.get_layer(name).output for name in layer_names]

    extract_model = tf.keras.Model(inputs=contract_model.input, outputs=layers)
    input = tf.keras.layers.Input(shape=input_shape)
    output_layers = extract_model(inputs=input)

    last_layer = output_layers[-1]


    feature_map = last_layer
    print(feature_map.shape)
    pooling_1 = global_average_pooling(feature_map.shape, 1)(feature_map)
    pooling_2 = global_average_pooling(feature_map.shape, 2)(feature_map)
    pooling_3 = global_average_pooling(feature_map.shape, 3)(feature_map)
    pooling_4 = global_average_pooling(feature_map.shape, 6)(feature_map)

    x = tf.keras.layers.Concatenate(axis=-1, name='psp_out')([pooling_1, pooling_2, pooling_3, pooling_4])
    x = Conv2D(filter_size*4, (1, 1), padding="same", name='after_psp')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([x, output_layers[3]])

    x = Conv2D(filter_size*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filter_size*4, 4, (2, 2), padding="same", name='up1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[2]])

    x = Conv2D(filter_size*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size*4, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filter_size*4, 4, (2, 2), padding="same", name='up2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[1]])

    x = Conv2D(filter_size*2, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size*2, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(filter_size*2, 4, (2, 2), padding="same", name='up3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[0]])

    x = Conv2D(filter_size, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filter_size, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(n_classes, (1, 1), activation="softmax", dtype='float32')(x)

    return tf.keras.Model(inputs=input, outputs=x)


if __name__ == "__main__":
    model = pspunet((512, 512, 3), 2)

    x = tf.random.uniform([1, 512, 512, 3], 0, 1)
    y = model(x)
    #print(y.shape)

    model.summary()