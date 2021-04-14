import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
from colour.plotting import render
import json
import requests


def saveCIE (xy, id1):
    #plotting x,y
    plot_chromaticity_diagram_CIE1931(standalone=False)

    # Plotting the *CIE xy* chromaticity coordinates
    x, y = xy
    plt.plot(x, y, 'o-', color='white')
    
    # Annotating the plot.
    plt.annotate('FEM',
                 xy=xy,
                 xytext=(-50, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))

    # Displaying the plot.
    render(
    filename = '%s.png' %id1,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True)

    #standalone=True,

def dominant (img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant


url1 = "http://173.248.157.26/arastu/api/single.php?id=2"
response = requests.get(url1)

data = response.json()
image_id = data['id']
url2 = data['image']
response = requests.get(url2)

with open(r'%s.jpg' %image_id,'wb') as f:
    f.write(response.content)

# Open the image. 
img_bgr = cv2.imread('%s.jpg' %image_id)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#Apply gamma correction.
gamma = 2.4
gamma_img = np.array(255*(img / 255) ** gamma, dtype = 'uint8')

avg_color_gamma = gamma_img.mean(axis=0).mean(axis=0)
dominant_color_gamma = dominant(gamma_img)

XYZ1 = colour.sRGB_to_XYZ(dominant_color_gamma)


xy =  colour.XYZ_to_xy(XYZ1)

saveCIE(xy, image_id)

url = 'https://www.gyanvihar.org/arastu/uploads/'
files = {'image': open('%s.png' %image_id, 'rb')}
requests.post(url, files=files)

