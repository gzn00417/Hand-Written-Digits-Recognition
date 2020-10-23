import numpy as np
import PIL
from torchvision import transforms
from torch.autograd import Variable
import pandas as pd


def get_image(image_path):
    return PIL.Image.open(image_path).convert("L").resize((28, 28))


def image_to_vector(image):
    return np.array(image).ravel()  # 转换成1*784的向量


def image_to_tensor(image):
    return transforms.ToTensor()(image).unsqueeze_(0)


def image_color_invert(image):
    return PIL.ImageOps.invert(image)


def process_pred(pred):
    pred = np.exp(pred)
    pred = (pred - pred.mean()) / pred.std()
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = pred / pred.sum()
    pred = pred.round(6)
    return pred.reshape(10, 1)


def predict(model, image):
    pred = model(Variable(image_to_tensor(image_color_invert(image)))).data.numpy()
    return pd.DataFrame(process_pred(pred), columns=["Prob"])
