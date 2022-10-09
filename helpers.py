from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
import numpy as np
import torch

def load_image(img_path, max_size=400, shape=None):
    """
    Loads image in and makes sure its of a specific size

    Parameters
    ----------
    img_path : String
        A path that leads to an image.
    max_size : Int, optional
        Max size of image, image bigger will be scaled down. The default is 
        400.
    shape : Tuple, optional
        A tuple containing the shape of the desired output. The default is 
        None.

    Returns
    -------
    image : Tensor
        Image Tensor with transforms and normalization applied.

    """
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    image_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

    image = image_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


def image_convert(image_tensor):
    """
    Converts a image tensor into a numpy array and unnormalize it. This is done
    to make it easy to display using matplotlib.

    Parameters
    ----------
    image_tensor : Tensor
        Image to unnormakize and convert.

    Returns
    -------
    image : nparray
        Converted array.

    """
    
    image = image_tensor.to("cpu").clone().detach().numpy().squeeze()
    
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """
    Grabs style and content features of image as its passed through the model. 
    Layers specified are for the VGG network seen in Gatys et al (2016).

    Parameters
    ----------
    image : Tensor
        Image to pass through.
    model : Torch Model
        DESCRIPTION.
    layers : Dict, optional
        The layers to grab style and features from. The default is layers
        from Gatys et al (2016).

    Returns
    -------
    features : Dict
        The collected features from the specified layers.
        
    """
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', # for content
                  '28': 'conv5_1'}
        
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x # creates dict of features
            
    return features


def gram_matrix(tensor):
    """
    Calculates the gram maxtrix given a tensor. Used in style transfer.  

    Parameters
    ----------
    tensor : Tensor
        Tensor to calculate gram matrix from.

    Returns
    -------
    gram : Tensor
        Tensor containg gram matrix

    """
    batch_size, d, h, w = tensor.size() # In this case batch_size is = 1
    tensor = tensor.view(batch_size * d, h * w)

    # calculate gram matrix
    gram = torch.mm(tensor, tensor.T)
    
    return gram 