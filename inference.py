import onnx
import caffe2.python.onnx.backend as backend
from skimage import io
import os
import numpy as np

net = onnx.load("./checkpoint/b030520c-e11d-11e8-a036-02665212c25a.proto")
rep = backend.prepare(net, device="CPU")
infer_data_folder = "./data/infer_data/"

def infer():
    for l in os.listdir(infer_data_folder):
        image_data = np.asfarray(io.imread(infer_data_folder + str(l)), dtype='float32')
        image_data.shape = (1, 1, 28, 28)
        outputs = rep.run(image_data)[0]
        predicted = outputs.argmax()
        print("We predicted: %s, Original was: %s" % (str(predicted), str(l.split('.')[0])))


if __name__ == '__main__':
    infer()
