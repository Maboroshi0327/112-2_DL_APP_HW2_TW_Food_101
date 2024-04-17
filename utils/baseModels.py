from tensorflow import keras


def DenseNet121() -> keras.Model:
    denseNet = keras.applications.densenet.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(150, 150, 3)
    )

    flatten = keras.layers.Flatten()(denseNet.output)
    dropout1 = keras.layers.Dropout(0.5)(flatten)
    dense1 = keras.layers.Dense(4096, activation="relu")(dropout1)
    dropout2 = keras.layers.Dropout(0.5)(dense1)
    output = keras.layers.Dense(101, activation="softmax")(dropout2)

    model = keras.Model(inputs=denseNet.input, outputs=output)

    return model


def EfficientNetB3() -> keras.Model:
    # # Flatten
    # efficientnet = keras.applications.efficientnet.EfficientNetB3(
    #     include_top=False, weights="imagenet", input_shape=(150, 150, 3)
    # )
    # flatten = keras.layers.Flatten()(efficientnet.output)
    # drop = keras.layers.Dropout(rate=0.5)(flatten)
    # output = keras.layers.Dense(101, activation="softmax")(drop)
    # model = keras.Model(inputs=efficientnet.inputs, outputs=output)

    # Global Average Pooling
    efficientnet = keras.applications.efficientnet.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(150, 150, 3), pooling="avg"
    )
    drop = keras.layers.Dropout(rate=0.5)(efficientnet.output)
    output = keras.layers.Dense(101, activation="softmax")(drop)
    model = keras.Model(inputs=efficientnet.inputs, outputs=output)

    return model


if __name__ == "__main__":
    model = DenseNet121()
    model = EfficientNetB3()
