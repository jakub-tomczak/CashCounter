from __future__ import division

from skimage import io, filters, feature, transform, draw,exposure
import skimage.morphology as mp
import skimage.measure as measure
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.measure import regionprops
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float, img_as_ubyte
from skimage.draw import circle_perimeter

from pylab import *
import matplotlib.patches as mpatches
import numpy as np
from ipykernel.pylab.backend_inline import flush_figures

'''
http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
https://github.com/astroluj/VC_OpenCV_Coin_Detection
http://www.emgu.com/wiki/files/1.4.0.0/html/0ac8f298-48bc-3eee-88b7-d2deed2a285d.htm
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html
'''

class coin:
    file_extension = "jpg"
    in_dir = "data/"
    out_dir = "out/"
    def __init__(self, img):
        self.img_oryg = img
        self.img_contoured = []
        self.segments = []
        self.img_size = 0

    def read_image(self):
        pass
    def find_contours(self, f):
        img_contoured = f(self.img_oryg)



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

def region_detector(img,t):

    binary = (img > t) * 255
    binary = np.uint8(binary)
    binary = mp.dilation(binary)

    binary = mp.label(binary)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(binary)

    for region in regionprops(binary):
        if region.area >= 1500:
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
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

path = ""
extension = "'"
def readImage(name):
    #filename = "%03d" % name
    file = "{path}/{name}.{extension}".format(path = path, name = name, extension = extension)
    print("Trying to open", file)
    return  io.imread(file)

def canny(img, name):
    img2 = mp.dilation(feature.canny(img, 4))
    #drawPlot(img2, img, name)
    return img2

def circle_detector(x, y, num):
    xc = np.mean(x)
    yc = np.mean(y)
    r = (x-xc)**2 + (y-yc)**2 #try to figure out radius of the circle with the middle of the xc and yc

    #circle estimator
    #plt.Circle((xc,yc), 20, color='k')
    #xy = measure.CircleModel().predict_xy(y, params=(2, 3, 4))

    #r = measure.CircleModel()
    #return r.estimate(np.array( tuple(zip(x,y)) ))

    #data = [num, np.mean(r), np.std(r), 100.jpg*np.std(r)/np.mean(r), "% ", np.percentile(r, 75) - np.percentile(r, 25), xc, yc]
    #return (" ".join(map(str, data)), np.std(r)/np.mean(r))
    return ("%.2f" % (100 * np.std(r) / np.mean(r)), ( np.std(r) / np.mean(r)))

    #verbose version
    #print(num, np.mean(r), np.std(r), 100.jpg*np.std(r)/np.mean(r), "% ", np.percentile(r, 75) - np.percentile(r, 25))

#uses find contours and percentile threshold

def findContours(img, number):
    img_size = len(img)*len(img[0])
    contours = measure.find_contours(img, .8, fully_connected='high')
    sizes = [len(contour) for contour in contours]
    print(np.mean(sizes), np.percentile(sizes, 75) - np.percentile(sizes, 25))
    perc = 25
    perMin = np.percentile(sizes, perc)
    perMax = np.percentile(sizes, 100 - perc)
    #remove using percentiles
    #cleaned = [contour for contour in contours if ( len(contour) > perMin and len(contour) < perMax)]
    #remove using possible shape surface
    cleaned = [contour for contour in contours if ( (np.max(contour[:,1])-np.min(contour[:,1])) * (np.max(contour[:,0])-np.min(contour[:,0])) > img_size*0.01 )]

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    iter = 0
    for n, contour in enumerate(cleaned):

        fig, ax = plt.subplots()
        #y
        perc1 = int(np.percentile(contour[:,0], 70))
        #x
        perc2 = int(np.percentile(contour[:,1], 30))
        contour = flip_pixels(contour, (perc2, perc1))

        text, percent = circle_detector(contour[:, 1], contour[:, 0], iter)

        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        patch = plt.Circle((int(np.mean(contour[:,1])), int(np.mean(contour[:,0]))), 2, color='r')
        ax.add_patch(patch)
        ax.text(int(np.min(contour[:,1])), int(np.min(contour[:,0])), text ,color = 'r' , fontsize=8)
        #rect = plt.Rectangle()
        #plt.show()
        #flush_figures()
        #plt.savefig("out/{}_detected.png".format(iter))
        break

        iter += 1
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    #plt.savefig("out/{}_detected.png".format(number))

def flip_pixels(contours, mean_point):
    new_contours = np.zeros((2*len(contours), 2))
    iter = 0
    for (x,y) in contours:
        new_contours[iter] = [x, y]
        iter+=1

    m_x = mean_point[0]
    m_y = mean_point[1]

    for (x,y) in contours:
        to_append = [m_x - (x-m_x), m_y - (y - m_y)]
        new_contours[iter] = to_append
        iter+=1
    return new_contours

def main():
    global path
    path = "data/old"
    global extension
    extension = "png"
    #img = readImage(2)
    #img = rgb2gray(img)
#    find(img)
    #[canny(rgb2gray(readImage(i)), "{}.png".format(i)) for i in range(1,5)]
    imgs_oryg = [readImage(i) for i in range(2, 3)]
    imgs = [canny(rgb2gray(imgs_oryg[i]), "out/old_{}.png".format(i)) for i in range(0, len(imgs_oryg))]
    #customized circles detector
    [findContours(imgs[i], i+1) for i in range(0, len(imgs))]

    #region detector with threshold
    #[region_detector(img, .7) for img in imgs]

    #hough circles
    #[hough_circles_detector(img, oryg) for img, oryg in zip(imgs, imgs_oryg)]


if __name__ == "__main__":
    main()
