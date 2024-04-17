import visualkeras
import tensorflow as tf
import tensorboard


def fn_visualkeras():
    model = tf.keras.models.load_model("./modelFiles_DenseNet121_V4/model_67.keras")
    visualkeras.layered_view(model, to_file="output.png")


def fn_tensorboard():
    print(tensorboard.__version__)


if __name__ == "__main__":
    fn_visualkeras()
