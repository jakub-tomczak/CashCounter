from __future__ import division
from pylab import *
from skimage import io, filters
import skimage.morphology as mp
import skimage.measure as measure
from skimage.color import rgb2gray

import numpy as np
from ipykernel.pylab.backend_inline import flush_figures

'''
http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
https://github.com/astroluj/VC_OpenCV_Coin_Detection
http://www.emgu.com/wiki/files/1.4.0.0/html/0ac8f298-48bc-3eee-88b7-d2deed2a285d.htm
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
'''

def hsv2rgb(h, s, v):
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
    return [r*255, g*255, b*255]

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

def contourDetector(image, threshold = .15):
    gray = rgb2gray(image)
    sobelImage = filters.sobel(gray)
    thresold = mp.dilation(colorThreshold(sobelImage, getMax(sobelImage)*threshold))

    #labeling contours
    #get labeled contours and num as a number of contours
    labeled, num = measure.label(thresold, background=0, neighbors=8, return_num=True)
    imSize = len(labeled) * len(labeled[0])

    #unique - list of contours id
    unique, counts = np.unique(labeled, return_counts=True)
    values = dict(zip(unique, counts))
    #remove contours that have less than .05% of total image pixels
    labeled = [[ (values[value] > imSize*0.0005)*value for value in row] for row in labeled]

    #finding centers
    #iterate over all contours and find centers of contours
    labeled = findCenters(labeled, values)

    #rewrite labeled values colored with hsv2rgb
    for row in range(0, len(labeled)):
        for i in range(0, len(labeled[row])):
            if labeled[row][i] > 0: # ommit background
                    image[row][i] = hsv2rgb(lerp(120, 360 , labeled[row][i]/(num+1)), 1, 1)    #num+1 since we have another color for point

    return image

def processAll(threshold = .15):
    lastFileIndex = 20
    images = [readImage("samolot%02d" % i) for i in range(0, lastFileIndex+ 1)]
    contours = [contourDetector(image) for image in images]
    displaySaveImage(contours, "planes_t{}.png".format(threshold))


def processOne(number):
    filename = "1" #"samolot%02d" % number
    image = readImage(filename)
    contour = contourDetector(image)
    displaySaveImage([contour], filename, resolution=100)

def colorThreshold(image, t):
    processed = (image > t) * 1
    flush_figures()
    return processed

def getMax(image):
    maxV = max(np.max(image, axis=1))
    return maxV

def readImage(name):
    return  io.imread("data/{img}.jpg".format(img = name))

def displaySaveImage(imgs, filename = "planes.png", resolution = 500):
    fig = figure(figsize=(20,20))
    if len(imgs) == 1:
        rows = 1
    else:
        rows = int(len(imgs)/2 +1)
    for i in range(0, len(imgs)):
        subplot(rows, 2, i+1)
        io.imshow(imgs[i])
    fig.savefig("out/"+filename, dpi=resolution)
def main():
    #processAll()
    processOne(1)
    #[processOne(num) for num in range(0,21)]

if __name__ == "__main__":
    main()
