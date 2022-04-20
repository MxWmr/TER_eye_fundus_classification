import numpy as np
import cv2 
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import pandas as pd 
from tqdm import tqdm 


def clustering(img):
    Z = img.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('segmented.png',res2)
    return res2,label

def find_disc_cup(res2,label):
    #### find the label of the optic disc and cup
    optic_cup=np.zeros([601,601])
    optic_disc=np.zeros([601,601])
    l1,l2,l3,l0 = None,None,None,None
    for i,row in enumerate(res2):
        for j,pix in enumerate(row):
            if label[601*i+j]==0 and l0 is None:
                l0=pix
            elif label[601*i+j]==1 and l1 is None:
                l1=pix
            elif label[601*i+j]==2 and l2 is None:
                l2=pix
            elif label[601*i+j]==3 and l3 is None:
                l3=pix
    L=[l0,l1,l2,l3]
    val_cup=max(L)
    L.remove(val_cup)
    val_disc=max(L)

    #### isolate disc and cup
    for i,row in enumerate(res2):
        for j,pix in enumerate(row):
            if pix==val_cup:
                optic_cup[i,j]=255
            elif pix==val_disc:
                optic_disc[i,j]=255

    plt.imshow(optic_cup, interpolation='nearest')
    plt.savefig('cup_segmented.png')
    plt.show()
    plt.imshow(optic_disc, interpolation='nearest')
    plt.savefig('disc_segmented.png')
    plt.show()
    
    return optic_cup,optic_disc


def find_cdr(optic_cup,optic_disc,draw):
    hough_radii_disc = np.arange(180, 300, 5)
    hough_radii_cup = np.arange(100, 250, 5)
    hough_res_disc = hough_circle(optic_disc, hough_radii_disc)
    hough_res_cup = hough_circle(optic_cup, hough_radii_cup)
    
    # Select the most prominent circle
    accums_d, cx_disc, cy_disc, radii_disc = hough_circle_peaks(hough_res_disc, hough_radii_disc,total_num_peaks=1)
    accums_c, cx_cup, cy_cup, radii_cup = hough_circle_peaks(hough_res_cup, hough_radii_cup,total_num_peaks=1)
    
    if radii_disc[0]!=0 and radii_cup[0]!=0:
        cdr=radii_cup[0]/radii_disc[0]
    else:
        cdr=0
        print('oh')
    
    if draw == True:
        draw_circles(img,cy_disc,cx_disc,radii_disc,radii_cup,cx_cup,cy_cup)


    return cdr

def draw_circles(img,cy_disc,cx_disc,radii_disc,radii_cup,cx_cup,cy_cup):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(img)

    circy_disc, circx_disc = circle_perimeter(cy_disc[0], cx_disc[0], radii_disc[0],shape=image.shape)
    image[circy_disc, circx_disc] = (20, 220, 20)

    circy_cup, circx_cup = circle_perimeter(cy_cup[0], cx_cup[0], radii_cup[0],shape=image.shape)
    image[circy_cup, circx_cup] = (20, 20, 220)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()



img = cv2.imread('reduced_dataset/train/glaucoma_positive/228.jpg')
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res2,label = clustering(img)
optic_cup,optic_disc = find_disc_cup(res2,label)
#cdr=find_cdr(optic_cup,optic_disc,True)
