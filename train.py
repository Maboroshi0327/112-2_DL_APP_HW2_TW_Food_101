from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import ImageFile
import warnings

from utils.baseModels import DenseNet121, EfficientNetB3
from utils.dataset import twFood101


class saveModel(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"./modelFiles/model_{epoch}.keras")


def main(batch_size=8, epochs=10):
    # Data
    dataset = twFood101()
    trainGenerator = dataset.trainGenerator(batch_size=batch_size)
    validGenerator = dataset.validGenerator(batch_size=batch_size)

    # Model
    model = DenseNet121()

    # Train
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00002122),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        trainGenerator,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=validGenerator,
        validation_batch_size=batch_size,
        callbacks=[saveModel()],
    )

    # Plot History
    acc_history = history.history["accuracy"]
    fig = plt.figure()
    (acc,) = plt.plot(acc_history)
    plt.title(f"Accuracy")
    plt.xlabel("epoch")
    plt.legend([acc], [f"Accuracy {acc_history[-1]}"], loc="lower right")
    plt.savefig(f"Accuracy.png")
    plt.close(fig)

    loss_history = history.history["loss"]
    fig = plt.figure()
    (loss,) = plt.plot(loss_history)
    plt.title(f"Loss")
    plt.xlabel("epoch")
    plt.legend([loss], [f"Loss {loss_history[-1]}"], loc="lower right")
    plt.savefig(f"Loss.png")
    plt.close(fig)


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    warnings.filterwarnings("ignore")
    main(batch_size=45, epochs=100)
