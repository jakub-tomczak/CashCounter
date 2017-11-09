import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, io
from skimage.color import rgb2gray
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.filters import sobel
import skimage.measure as measure
import skimage.morphology as mp
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

def preprocessImage(tempImage):
    #image = tempImage**1.5
    return 0#image

def processImage(imgNumber = -1):
    if imgNumber > -1:
        greyimage = rgb2gray(io.imread("data/{img}.JPG".format(img=imgNumber)))
        greyimage = greyimage > .5
    else:
        greyimage = data.coins()


    image = img_as_ubyte(greyimage)
    edges = sobel(image)
    edges =  mp.dilation(canny(image, sigma=3, low_threshold=30, high_threshold=60))
    drawPlot(edges)

    # Detect two radii
    hough_radii = np.arange(100, 250, 20)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=20)

    # Draw them
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image[circy, circx] = (220, 20, 20)

    drawPlot(image)



def drawPlot(image):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()



def main():
    processImage(1)

if __name__ == "__main__":
    main()
