#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 21:29:13 2019

@author: paulinakwasniewska
"""

import numpy as np
import os 
from matplotlib import pyplot as plt
from skimage import io
from skimage import color
from scipy import ndimage as nd
from scipy import signal

#==============================================================================
# Zaimplementuj fuckcję przyjmującą jako argument obraz 2-D oraz łańcuch 
# określajścy typ gradientu. 
# Zwróć gradient w osi X, w osi Y, magnitudę oraz kąt gradientu.
#==============================================================================

def our_gradient(image, mode):
    gradient_x = np.zeros(image.shape)
    gradient_y = np.zeros(image.shape)
    if mode == 'Forward':
        gradient_x[:, 0:-1] = image[:, 1:] - image[:, 0:-1]
        gradient_y[0:-1, :] = image[1:, :] - image[0:-1, :]
    elif mode == 'Central':
        gradient_x[:, 1:-1] = image[:, 2:] - image[:, 0:-2]
        gradient_y[1:-1, :] = image[2:, :] - image[0:-2, :]   
    elif mode == 'Backward':
        gradient_x[:, 1:] = image[:, 1:] - image[:, 0:-1]
        gradient_y[1:, :] = image[1:, :] - image[0:-1, :]
    else:
        raise ValueError("Invalid mode.")
    
    gradient_angle = np.arctan2(gradient_y, gradient_x)
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return [gradient_x, gradient_y, gradient_magnitude, gradient_angle]


current_path = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(current_path, 'xray.jpg')
image = io.imread(image_path)
image = color.rgb2gray(image)

gradients  = our_gradient(image, 'Central')
gradient_x, gradient_y = gradients[0], gradients[1]
gradient_magnitude = gradients[2]
gradient_angle = gradients[3]

plt.figure(1)
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(gradient_x, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(gradient_y, cmap='gray')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(gradient_magnitude, cmap='gray', vmin=0, vmax=0.1)
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(gradient_angle, cmap='gray')
plt.axis('off')
plt.show()

#==============================================================================
# Zaimplementuj funkcje ̨, która przyjmuje obraz, a zwraca obraz znormalizowany 
# do zakresu [0-1].
#==============================================================================

def normalize(image):
    lmin = np.min(image)
    lmax = np.max(image)
    normalized_image = (image - lmin) / (lmax - lmin) 
    return normalized_image

normalized_image = normalize(image)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')

#==============================================================================
# Zaimplementuj funkcje ̨, która jako argument przyjmuje obraz 2-D oraz filtr 
# symetryczny reprezentowany przez dwa wektory jednowymiarowe. Funkcja powinna 
# zwracac ́ obraz po filtracji.
#==============================================================================

#==============================================================================
#  Zaimplementuj funkcje ̨ licząca ̨ Laplasjan obrazu 2-D.
#==============================================================================
def laplasjan(image):
    laplasjan = np.array([
            [0, 1, 0],
            [1, -4, 1], 
            [0, 1, 1]
        ])
    image_laplasjan = signal.convolve2d(image, laplasjan)
    return image_laplasjan

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(laplasjan(image), cmap='gray')
plt.axis('off')

#==============================================================================
# Zaimplementuj funkcje, która jako argument przyjmuje obraz oraz pożądana ̨ 
# wielkość filtru medianowego. Funkcja powinna zwracac ́ obraz po filtracji 
# medianowej.
#==============================================================================
def median_filter(image, size):
    image_median_filter = nd.generic_filter(image, np.median, 
                                            footprint = np.ones((size,size)))
    return image_median_filter

plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(median_filter(image, 5), cmap='gray')
plt.axis('off')

