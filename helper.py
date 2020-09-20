#Helper functions for neural style transfer


import torch
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from torch import optim
from torchvision import transforms 


def load_image(img_path, max_size=400, size=None):
    ''' 
    Load an image from path and perform transforms on it.
    
    params:
        img_path (str): path/url to image file
        max_size (int): maximum size of image - smaller is faster
        
    returns:
        img (tensor): pytorch tensor of processed image 
    '''
    #get image
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    #set size
    size = size or max(image.size)
    img_size = min(size, max_size)
    
    #create and apply transforms
    img_transform = transforms.Compose([
                        transforms.Resize(img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    image = img_transform(image)[:3,:,:].unsqueeze(0)
    
    return image
    

def to_img(tensor):
    """ 
    Convert a tensor to rgb image.
    """
    image = tensor.to("cpu").clone().detach() 
    image = image.numpy().squeeze() #remove batch dimension
    image = image.transpose(1,2,0)  #transpose to set color channel
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406)) #unnormalise :)
    image = image.clip(0, 1) #prevent negatives

    return image
    
    
def gram_matrix(tensor):
    """ 
    Calculate the Gram Matrix of a tensor 
        
    """
    tensor = tensor.view(tensor.size(1), tensor.size(2)*tensor.size(3))
    
    return torch.mm(tensor , tensor.T)
    
    
def forwardpass(image, model, layers=None):
    """ 
        Perform forward pass of an image through a model and extract selected layers. 
        
        params:
            image (tensor): input image
            model (nn.module): neural network
            layers (dict): dictionary of layer index and module names
    """

    ## layers to extract the content and style representations of image
    if layers is None:
        layers = {'0': 'conv1_1', '5':'conv2_1', '10': 'conv3_1', '19':'conv4_1', '21':'conv4_2', '28':'conv5_1'}

    outputs = {}
    
    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            outputs[layers[name]] = image
            
    return outputs

def show_results(content, style, target):
  # display content and final, target image
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
  ax1.imshow(to_img(content))
  ax2.imshow(to_img(style))
  ax3.imshow(to_img(target))

def style_transfer(content, style, model, style_weights, optimizer=optim.Adam, lr=0.003, epochs=5000, 
                   show_every=5000, content_weight = 1, style_weight = 1e6, device='cpu', target=None):
  '''
  Train a target image to embody the details of content and the style 
  '''

  model=model.to(device)
  content=content.to(device)
  style=style.to(device)
  target = target or content.clone().requires_grad_(True)
  target = target.to(device)
  print('Target image initialised!')

  content_output = forwardpass(content, model)
  style_output = forwardpass(style, model)

  print('Style and content outputs initialised!')

  style_grams = {layer: gram_matrix(style_output[layer]) for layer in style_output}

  optimizer = optimizer([target], lr=lr)

  print('Training started...')
  for epoch in range(epochs):
        
    target_output = forwardpass(target,model)

    #calculate loss for content
    content_loss = torch.mean((target_output['conv3_1'] - content_output['conv3_1'])**2)
    
    # calculate loss for style
    style_loss = 0
    
    for layer in style_weights:
        
        output = target_output[layer]
        _, channel, height, width = output.shape

        target_gram = gram_matrix(output)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)       
        
        style_loss += layer_style_loss / (channel * height * width)  
        
    ## Compute total loss
    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    # update target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # display intermediate images and print the loss
    if epoch % show_every == 0:
        print(f'Epoch {epoch} Total loss: {total_loss.item()}')
        plt.imshow(to_img(target))
        plt.show()
  
  print('Style transfer complete!')

  return target