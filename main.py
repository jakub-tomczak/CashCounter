from __future__ import division
from pylab import *
from skimage import io, filters, feature, transform, draw
import skimage.morphology as mp
import skimage.measure as measure
import matplotlib.patches as mpatches
from skimage.color import rgb2gray

import numpy as np
from ipykernel.pylab.backend_inline import flush_figures
from skimage.measure import regionprops

'''
http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
https://github.com/astroluj/VC_OpenCV_Coin_Detection
http://www.emgu.com/wiki/files/1.4.0.0/html/0ac8f298-48bc-3eee-88b7-d2deed2a285d.htm
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
'''

def hsv2rgb(h, s, v, alpha = False):
    import math
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    if alpha:
        return [r*255, g*255, b*255, 255]
    return [r*255, g*255, b*255]

def drawPlot(image):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()
def lerp(start, stop, value):
    return start + (stop - start)*value

def findCenters(labeled, values):
    xs = np.zeros(len(values))
    ys = np.zeros(len(values))
    count = np.zeros(len(values))

    for row in range(0,len(labeled)):
        for col in range(0,len(labeled[row])):
            for key in values:
                if key == 0:
                    continue    #background number
                if labeled[row][col] == key:
                    xs[key] +=  col
                    ys[key] += row
                    count[key] +=1


    for i in range(0, len(xs)):
        if count[i] == 0:
            continue

        x = int(xs[i] / count[i])
        y = int(ys[i] / count[i])
        print(i, x , y, len(labeled)*len(labeled[0]))


        for row in range(y-6, y+6):
            for col in range(x-6, x+6):
                labeled[row][col] = len(values)

    return labeled



def detector_canny(image):
    gray = rgb2gray(image)

    canny = feature.canny(gray, sigma=1.4)
    labeled, num = measure.label(canny, return_num = True)

    unique, counts = np.unique(labeled, return_counts=True)
    values = dict(zip(unique, counts))
    imSize = len(labeled) * len(labeled[0])
    labeled = [[ (values[value] > imSize*0.0005)*value for value in row] for row in labeled]

    for row in range(0, len(labeled)):
        for i in range(0, len(labeled[row])):
            if labeled[row][i] > 0: # ommit background
                    image[row][i] = hsv2rgb(lerp(120, 360 , labeled[row][i]/num), 1, 1, len(image[row][i]) > 3)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for region in regionprops(labeled):

        minr, minc, maxr, maxc = region.bbox
        if abs((maxr-minr) - (maxc-minc)) > 20:
            continue
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


    #drawPlot(image)

def processOne(number):

    image = readImage(number)
    contour = detector_canny(image)


#porównać histogramy 1 i 2
def colorThreshold(image, t):
    processed = image > t
    flush_figures()
    return processed

def getMax(image):
    maxV = max(np.max(image, axis=1))
    return maxV

def readImage(name):
    return  io.imread("data/old/{img}.png".format(img = name))

def main():
    processOne(3)
    #[processOne(i) for i in range(1,5)]

if __name__ == "__main__":
    main()
