from CNN import Classifier

import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.color import rgb2gray
from skimage.transform import resize

def threshold_image(image):
    """
    Returns an binary image
    """
    image[image > image.mean()] = 1
    image[image <= image.mean()] = 0
    return image


def predict_on_single_image(net, img):
    """ Plots image and gives prediction"""
    
    # Make sure all imgs are same format
    try:
        img = img.resize(1, 45, 45).unsqueeze(0)
    except:
        print("Unaccepted shape for image")
    
    output = net(img)
    pred = (F.softmax(output).data).max(1)[1]
    return pred.item()
    
  
def predict_image(image, weights):
    """
    Takes in an image, processes it and uses weights for CNN
    Returns a prediction within range 0-14
    """
    
    net = Classifier()
    net.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    
    image = image.resize((45,45), Image.ANTIALIAS)
    image = PIL.ImageOps.invert(image)
    image = np.asarray(image)
    image = rgb2gray(image)

    image = threshold_image(image)

    image = torch.from_numpy(image.astype(np.float32))
    image = transforms.functional.normalize(image.unsqueeze(0), 0.1307, 0.3081)
    
    return predict_on_single_image(net, image)
    