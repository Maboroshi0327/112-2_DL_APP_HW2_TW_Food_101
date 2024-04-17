import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from skimage import transform

from utils.dataset import twFood101


def load_img(filename, target_w=150, target_h=150):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255.0
    np_image = transform.resize(np_image, (target_w, target_h, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


def test1():
    test_dict = {}
    for root, dirs, files in os.walk("./tw_food_101/test"):
        for filename in files:
            test_id, file_ext = os.path.splitext(filename)
            test_dict[test_id] = filename

    results = []
    model = tf.keras.models.load_model("./modelFiles/model_20.keras")
    for i in range(len(test_dict)):
        img = load_img("./tw_food_101/test/" + test_dict[str(i)], 150, 150)
        ret = model.predict(img, verbose=0)
        results.append(np.argmax(ret))
        print(i, "./tw_food_101/test/" + test_dict[str(i)])

    with open("pred_results.csv", "w") as f:
        f.write("Id,Category\n")
        for i in range(len(results)):
            f.write(str(i) + "," + str(results[i]) + "\n")


def test2():
    model = tf.keras.models.load_model("./modelFiles_DenseNet121_V4/model_67.keras")

    with open("tw_food_101_test_pred.csv", "w") as f1:
        f1.write("Id,Category\n")
        with open("./tw_food_101/tw_food_101_test_list.csv", "r") as f2:
            lines = f2.readlines()
            for line in lines:
                [index, path] = line.split("\n")[0].split(",")
                path = "./tw_food_101/" + path
                image = cv2.imread(path).astype("float32") / 255.0
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_NEAREST)
                image = np.expand_dims(image, axis=0)
                result = model.predict(image, verbose=0)[0]
                result = np.argmax(result, axis=-1)
                f1.write(str(index) + "," + str(result) + "\n")
                print(index, path)


if __name__ == "__main__":
    test2()
