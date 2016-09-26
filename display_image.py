import urllib
from PIL import Image
import matplotlib.pyplot as plt

def display_image(url_name,crop_in=100):
    '''
    A tool for plotting an SDSS GZ image from a given URL.
    
    Inputs:
    -------
    url_name: url name of the image to retrieve
    
    crop_in: number of pixels to crop in from both sides.
    *SDSS GZ images are 424 pixels by 424 pixels.
    '''
  
    # Get the url name:
    urllib.request.urlretrieve(url_name,"image.jpg")
    # Open -> crop -> display -> remove the image.
    im=Image.open("image.jpg")
    l=424 # Image size
    im=im.crop((crop_in,crop_in,l-crop_in,l-crop_in))
    plt.imshow(im)
    os.remove("image.jpg")
    plt.xticks([])
    plt.yticks([])
    return None