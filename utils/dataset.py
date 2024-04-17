import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


class twFood101:
    def __init__(
        self,
        root_dir="./tw_food_101",
        train_dir="/train",
        valid_dir="/valid",
    ):
        self.__root_dir = root_dir
        self.__train_dir = root_dir + train_dir
        self.__valid_dir = root_dir + valid_dir

    def trainGenerator(self, batch_size=8):
        gen = ImageDataGenerator(
            rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
        )
        generator = gen.flow_from_directory(
            self.__train_dir,
            target_size=(150, 150),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42,
        )
        return generator

    def validGenerator(self, batch_size=8):
        gen = ImageDataGenerator(rescale=1.0 / 255)
        generator = gen.flow_from_directory(
            self.__valid_dir,
            target_size=(150, 150),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
        )
        return generator


if __name__ == "__main__":
    dataset = twFood101()
    trainGenerator = dataset.trainGenerator()
    validGenerator = dataset.validGenerator()


# def generator():
#     for index, data in enumerate(gen):
#         if index > gen.__len__():
#             return
#         yield data

# dataset = tf.data.Dataset.from_generator(
#     generator,
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([None, 200, 300, 3], [None, 101]),
# )
