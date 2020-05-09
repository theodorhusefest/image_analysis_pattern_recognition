import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from skimage.transform import resize
from skimage.color import rgb2gray

def update_datasheet(image_dir = "./data/operators"):
    """ 
    Updates datasheet to fit current content of imagefolder 
    Filename must include correct label [plus, minus, mul, div, eq]
    """
    
    data = pd.DataFrame(columns = ["path", "label"])

    i = 0
    for file in os.listdir(image_dir):
        if "plus" in file:
            newrow = pd.DataFrame(columns = ["path", "label"],data = {"path": file, "label": 0}, index = [0])

        elif "minus" in file:
            newrow = pd.DataFrame(columns = ["path", "label"],data = {"path": file, "label": 1}, index = [0])

        elif "mul" in file:
            newrow = pd.DataFrame(columns = ["path", "label"],data = {"path": file, "label": 2}, index = [0])

        elif "div" in file:
            newrow = pd.DataFrame(columns = ["path", "label"],data = {"path": file, "label": 3}, index = [0])

        elif "eq" in file:
            newrow = pd.DataFrame(columns = ["path", "label"],data = {"path": file, "label": 4}, index = [0])

        data = data.append(newrow, ignore_index = True)
        i += 1

    data = data.reset_index(drop=True)
    data.to_csv(image_dir + "/datasheet.csv")
    
    
def get_operators_from_original(filepath = "./data/original_operators.png"):
    """
    Takes in filepath the original image with all operators
    Returns each operator in 32x32 and greyscale
    """
    operators = Image.open(filepath)
    img = np.asarray(operators)
    img_grey = rgb2gray(img)
    
    N = 316
    plus = img_grey[:, 0:N]
    equals = img_grey[:, N:2*N]
    min_offset = 75
    minus = img_grey[:, 2*N + min_offset :3*N + min_offset]
    div_offset = 125
    div = img_grey[:, 3*N + div_offset:4*N + div_offset]
    mul_offset = 155
    mul = img_grey[:, 4*N + mul_offset:-1]
    
    operators = [plus, minus, div, mul, equals]
    operators = [resize(operators[i], (32, 32)) for i in range(len(operators))]
    op_tensors = [torch.from_numpy(operators[i].astype(np.float32)) for i in range(len(operators))]

    
    return op_tensors