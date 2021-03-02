import numpy as np
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, \
    BatchNormalization, Activation, AveragePooling2D


def vgg16(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding="same", activation="relu", name="copy_crop1"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(128, (3, 3), padding="same", activation="relu", name="copy_crop2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu", name="copy_crop3"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu", name="copy_crop4"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="last_layer"))
    return model


def global_average_pooling(input, gap_size):
    w, h, c = (input.shape[1:])
    x = AveragePooling2D((w / gap_size, h / gap_size))(input)
    x = Conv2D(c // 4, (1, 1), padding="same", activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.image.resize(x, (w, h))
    return x


def contract_path(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(input)
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="copy_crop1")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="copy_crop2")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="copy_crop3")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", name="copy_crop4")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(1024, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(1024, (3, 3), padding="same", activation="relu", name="last_layer")(x)
    contract_path = tf.keras.Model(inputs=input, outputs=x)
    return contract_path


def pspunet(input_shape, n_classes):
    contract_model = contract_path(input_shape=input_shape)
    layer_names = ["copy_crop1", "copy_crop2", "copy_crop3", "copy_crop4"]
    layers = [contract_model.get_layer(name).output for name in layer_names]

    extract_model = tf.keras.Model(inputs=contract_model.input, outputs=layers)
    input = tf.keras.layers.Input(shape=input_shape)
    output_layers = extract_model(inputs=input)
    last_layer = output_layers[-1]

    feature_map = last_layer
    pooling_1 = global_average_pooling(feature_map, 1)
    pooling_2 = global_average_pooling(feature_map, 2)
    pooling_3 = global_average_pooling(feature_map, 3)
    pooling_4 = global_average_pooling(feature_map, 6)
    x = tf.keras.layers.Concatenate(axis=-1)([pooling_1, pooling_2, pooling_3, pooling_4])
    x = Conv2D(256, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = tf.keras.layers.Concatenate()([x, output_layers[3]])

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(256, 4, (2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[2]])

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, 4, (2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[1]])

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, 4, (2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.Concatenate()([x, output_layers[0]])

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #x = Conv2D(n_classes, (1, 1), activation="relu")(x)

    if n_classes == 1:
        x = Conv2D(n_classes, 1, padding='same', activation='sigmoid', name='Output')(x)
        print('Out Put with Sigmoid')
    else:
        x = Conv2D(n_classes, 1, padding='same', activation='softmax', name='Output')(x)
        print(x.shape)
        print('Out Put with Softmax')

    return tf.keras.Model(inputs=input, outputs=x)


if __name__ == "__main__":

    model = pspunet((512, 1024, 3), 5)

    x = tf.random.uniform([2, 512, 1024, 3], 0, 1)

    y = model(x)

    '''
    from util.func import *
    from util.dataloader.mapilaryVitas import *

    data_path = '/home/seungtaek/ssd1/datasets/mapillary_vistas'
    json_path = os.path.join(data_path, 'config_v2.0.json')
    print(json_path)
    Labels = pars_json_label(json_path)

    ignore_class = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29,
                    30, 31, 32, 33, -1]

    dataset = get_dataset(data_path, ignore_class, 512, 1024, batch_size=2)

    for i, data in enumerate(dataset['train']):
        display_list = []
        print(data[0][0].shape)
        print(data[1][0].shape)

        pred = model(data[0])
        pred = tf.argmax(pred, axis=-1)
        print(pred.shape)

        uniques, idx, counts = get_uniques(pred[0])

        tf.print("tf.shape(uniques) =", tf.shape(uniques))
        tf.print("tf.shape(idx) =", tf.shape(idx))
        tf.print("tf.shape(counts) =", tf.shape(counts))
        tf.print("uniques =", uniques)

        display_list.append(data[0][0])
        display_list.append(data[1][0])
        display_list.append(pred[0])
        display_sample_one(display_list)
        if i == 3: break

    display_sample(display_list)
    '''


    model.summary()