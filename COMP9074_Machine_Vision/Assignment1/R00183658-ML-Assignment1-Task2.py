import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt


def print_line():
    print('\n========================================================')
    

print_line()
print('\nTask 2 (17 points)')


######################################################################
#
#
#  PART 2A
#
#
######################################################################

print_line()
print('\nPart A (3 points)')

'''
[2 points]
Download the input image files Assignment_MV_01_image_1.jpg (the same as in the previous task) and Assignment_MV_01_image_2.jpg from Canvas.
Load both files and convert them into a single channel grey value image.

[1 point]
Make sure the data type is float32 to avoid any rounding errors.
'''
fname1 = 'Assignment_MV_01_image_1.jpg'
fname2 = 'Assignment_MV_01_image_2.jpg'

img1 = cv2.imread(fname1).astype(np.float32)
img2 = cv2.imread(fname2).astype(np.float32)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


######################################################################
#
#
#  PART 2B
#
#
######################################################################

print_line()
print('\nPart B (3 points)')

'''
[3 points]
The window on the 1st floor above the arch on the left wing is in a rectangle with the image coordinates ((360,210), (430,300)) in the first input image.

Draw a rectangle around this window in the input image and display it.
'''

'''
Define start and end points of rectangle.
'''
start_point = (360, 210)
end_point   = (430, 300) 

'''
Draw reactangle into picture and plot it.
'''
color = (255)
thickness = 2
col_rect = cv2.rectangle(img1, start_point, end_point, (color, color, color), thickness)
gray_rect = cv2.rectangle(gray1, start_point, end_point, color, thickness)

plt.figure(figsize=(20,10))
plt.title('Task 2B - Marked window in first image - Color', fontsize=16)
plt.imshow(col_rect/np.max(col_rect))
plt.show()

plt.figure(figsize=(20,10))
plt.title('Task 2B - Marked window in first image - Gray', fontsize=16)
plt.imshow(gray_rect, cmap='gray')
plt.show()

'''
[2 points]
Now cut out the image patch only containing the window and display it as image.
'''
'''
Create a crop of the window of interest.
Define width and height of crop image.
'''
x = start_point[0]
y = start_point[1]
w = end_point[0]-x
h = end_point[1]-y
crop = gray1[y:y+h, x:x+w].copy()

plt.figure(figsize=(10,5))
plt.title('Task 2B - Cropped Window - Gray', fontsize=16)
plt.imshow(crop, cmap='gray')
plt.show()


######################################################################
#
#
#  PART 2C
#
#
######################################################################

print_line()
print('\nPart C (11 points)')

'''
[2 points]
Calculate the mean and standard deviation of the cut-out patch from subtask B.
'''
crop_mean = crop.mean()
crop_stdev = crop.std()

print('\nCrop mean :', crop_mean)
print('Crop stdev:', crop_stdev)


'''
[2 points]
Go through all positions in the second input image and cut out a patch of equal size.
'''
def generate_patches(img):
    '''
    Iterate over image2 pixels (x, y) and cut out patches of the size of the cropped out window.
    In below implementation I accept that due to crop width and height the right and bottom borders are left "unprocessed".
    
    Potential solutions:
      - Padding
      - Moving to subscrops
      - ...
    
    Returns dictionary matching a patch to (x,y) coordinates.
    '''
    patches = {}
    #y = 0
    for x in range(0, img.shape[0] - h + 1):
        for y in range(0, img.shape[1] - w + 1):
            patches[(x,y)] = {}
            patches[(x,y)]['patch'] = gray2[x:x+h, y:y+w]
            
            assert patches[(x,y)]['patch'].shape[0] == 90, patches[(x,y)]['patch'].shape
            assert patches[(x,y)]['patch'].shape[1] == 70, patches[(x,y)]['patch'].shape
    
    return patches

print('\nGenerating patches... Patience please...')
patches = generate_patches(gray2)
print('Done')


'''
[3 points]
Also calculate mean and standard deviation and from this the cross-correlation between the two patches.
'''
def xcor(patch, patch_mean, patch_stdev, crop, crop_mean, crop_stdev):
    '''
    Calculate the cross-correlation between crop and patch.
    '''
    result = np.mean((crop - crop_mean) * (patch - patch_mean))/(crop_stdev*patch_stdev)
    return result

'''
For each patch of image2, calculate the mean, stdev and cross-correlation to crop image.
'''
print('\nCalculation cross-correlations... Patience please...')
for k, v in patches.items():
    v['mean'] = v['patch'].mean()
    v['stdev'] = v['patch'].std()
    v['xcor'] = xcor(v['patch'], v['mean'], v['stdev'], crop, crop_mean, crop_stdev)
print('Done')

'''
[2 points]
Create and display an image of all cross-correlations for all potential positions in the second image.
'''
'''
Create a new image array of same dimensions as gray2
'''
xcor_img = np.zeros((gray2.shape[0], gray2.shape[1]))

'''
Iterate over all patches and copy cross-correlation value into xcor_img pixels
Note: This creates an emtpy border where remaining width and height are less than the crop shape.
'''
for x in range(0, xcor_img.shape[0] - h + 1):
    for y in range(0, xcor_img.shape[1] - w + 1):
        xcor_img[x][y] = patches[(x,y)]['xcor']

plt.figure(figsize=(20,10))
plt.title('Task 2C - Image of possible cross-correlations', fontsize=16)
plt.imshow(xcor_img, cmap='gray')
plt.show()


'''
[2 points]
Find the position with maximum cross-correlation and draw a rectangle around this position in the second input image. Display the result.
'''
'''
np.argmax gives a scalar value representing the largest pixel in xcor_img.
Get pixel location by division and modulo operation.
'''
x_max = int(np.argmax(xcor_img) / 1024)
y_max = int(np.argmax(xcor_img) % 1024)
print('\nCoordinates of largest xcor value (x, y):', x_max, y_max)

'''
Define start_point from max pixel and end_point by adding crop width + height.
'''
start_point = (y_max, x_max)
end_point   = (y_max+w, x_max+h) 

'''
Draw reactangle into picture and plot it.
'''
color = (255)
thickness = 2
col2_rect = cv2.rectangle(img2, start_point, end_point, (color, color, color), thickness)
gray2_rect = cv2.rectangle(gray2, start_point, end_point, color, thickness)

plt.figure(figsize=(20,10))
plt.title('Task 2C - Marked window in second image - Color', fontsize=16)
plt.imshow(col2_rect/np.max(col2_rect))
plt.show()

plt.figure(figsize=(20,10))
plt.title('Task 2C - Marked window in second image - Gray', fontsize=16)
plt.imshow(gray2_rect, cmap='gray')
plt.show()

print_line()