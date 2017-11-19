from __future__ import division
from pylab import *
from skimage import io, filters, feature, transform, draw,exposure
import skimage.morphology as mp
import skimage.measure as measure
import matplotlib.patches as mpatches
import matplotlib.pyplot as graph
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte
from IPython.display import display

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

def drawPlot(image, name = "image.png"):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    plt.savefig(name)
    #plt.show()

def drawPlot(image, oryg_image, name = "image.png"):
    fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10,10))
    ax.imshow(image, cmap=plt.cm.gray)
    ax2.imshow(oryg_image, cmap=plt.cm.gray)
    plt.savefig(name)
    flush_figures()

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

def on_change(img,perc=0):
    MIN = np.percentile(img, perc)
    MAX = np.percentile(img, 100-perc)
    #Percentyl – kwantyl rzędu k/100, gdzie k=1, … , 99.
    #Intuicyjnie mówiąc, percentyl jest wielkością, poniżej której padają wartości zadanego procentu próbek.

    norm = (img - MIN) / (MAX - MIN)
    norm[norm[:,:] > 1] = 1
    norm[norm[:,:] < 0] = 0

    figure(figsize=(15,5))
    subplot(1,2,1); plt.imshow(norm, cmap='gray')
    subplot(1,2,2); plot_hist(norm)
    flush_figures()

def find(img,level=0.5):
    contours = measure.find_contours(img, level)
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)


    for n, contour in enumerate(contours):
        if len(contour) > 100:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def plot_hist(img):
    img = img_as_ubyte(img)
    histo, x = np.histogram(img, range(0, 256), density=True)
    a = plot(np.arange(0, 255, 1),histo)

    xlim(0, 255)

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

def canny(img, name):
    img2 = feature.canny(img, 4)
    #drawPlot(img2, img, name)
    return img2

def circle_detector(x, y, num):
    xc = np.mean(x)
    yc = np.mean(y)
    r = (x-xc)**2 + (y-yc)**2 #kiepski pomysł - zakłada środek w (0,0)
    #plt.Circle((xc,yc), 20, color='k')
    #xy = measure.CircleModel().predict_xy(y, params=(2, 3, 4))

    #r = measure.CircleModel()
    #return r.estimate(np.array( tuple(zip(x,y)) ))

    data = [num, np.mean(r), np.std(r), 100*np.std(r)/np.mean(r), "% ", np.percentile(r, 75) - np.percentile(r, 25), xc, yc]
    return (" ".join(map(str, data)), np.std(r)/np.mean(r))
    #print(num, np.mean(r), np.std(r), 100*np.std(r)/np.mean(r), "% ", np.percentile(r, 75) - np.percentile(r, 25))

#uses find contours and percentile threshold
def findContours(img):
    contours = measure.find_contours(img, .8, fully_connected='high')
    sizes = [len(contour) for contour in contours]
    perc = 10
    perMin = np.percentile(sizes, perc)
    perMax = np.percentile(sizes, 100 - perc)
    cleaned = [contour for contour in contours if ( len(contour) > perMin and len(contour) < perMax )]
    # Display the image and plot all contours found
    fig, ax = plt.subplots()

    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    iter = 0
    for n, contour in enumerate(contours):
        text, percent = circle_detector(contour[:, 1], contour[:, 0], iter)
        if percent > .75:
            continue
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        patch = plt.Circle((int(np.mean(contour[:,1])), int(np.mean(contour[:,0]))), .2, color='blue')
        ax.add_patch(patch)
        #ax.text(int(np.min(contour[:,1])), int(np.min(contour[:,0])), text , fontsize=8)
        #plt.savefig("testR/{}.png".format(iter))


        iter += 1
    #return 0
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def main():
    img = readImage(2)
    img = rgb2gray(img)
#    find(img)
    #[canny(rgb2gray(readImage(i)), "{}.png".format(i)) for i in range(1,5)]

    imgs = [canny(rgb2gray(readImage(i)), "{}.png".format(i)) for i in range(1, 5)]
    [findContours(image) for image in imgs]
    #on_change(img, 5)
    #processOne(3)
    #[processOne(i) for i in range(1,5)]

if __name__ == "__main__":
    main()
