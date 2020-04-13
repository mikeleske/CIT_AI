import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt


def print_line():
    print('\n========================================================')


print_line()
print('\nTask 1 (33 points)')

def plot_grid34(data, title):
    fig, axs = plt.subplots(4, 3, figsize=(15,20))
    fig.suptitle(title, fontsize=16)
    axs[0, 0].imshow(data[0], cmap='gray')
    axs[0, 1].imshow(data[1], cmap='gray')
    axs[0, 2].imshow(data[2], cmap='gray')
    axs[1, 0].imshow(data[3], cmap='gray')
    axs[1, 1].imshow(data[4], cmap='gray')
    axs[1, 2].imshow(data[5], cmap='gray')
    axs[2, 0].imshow(data[6], cmap='gray')
    axs[2, 1].imshow(data[7], cmap='gray')
    axs[2, 2].imshow(data[8], cmap='gray')
    axs[3, 0].imshow(data[9], cmap='gray')
    axs[3, 1].imshow(data[10], cmap='gray')
    axs[3, 2].imshow(data[11], cmap='gray')
    
    axs[0, 0].set_title('sigma=1.0')
    axs[0, 1].set_title('sigma=1.4142')
    axs[0, 2].set_title('sigma=2.0')
    axs[1, 0].set_title('sigma=2.8284')
    axs[1, 1].set_title('sigma=4.0')
    axs[1, 2].set_title('sigma=5.6568')
    axs[2, 0].set_title('sigma=8.0')
    axs[2, 1].set_title('sigma=11.3137')
    axs[2, 2].set_title('sigma=16.0')
    axs[3, 0].set_title('sigma=22.6274')
    axs[3, 1].set_title('sigma=32.0')
    axs[3, 2].set_title('sigma=45.2548')
    
    plt.show()

def plot_grid_dogs(data, title):
    fig, axs = plt.subplots(4, 3, figsize=(15,20))
    fig.suptitle(title, fontsize=16)
    axs[0, 0].imshow(data[0], cmap='gray')
    axs[0, 1].imshow(data[1], cmap='gray')
    axs[0, 2].imshow(data[2], cmap='gray')
    axs[1, 0].imshow(data[3], cmap='gray')
    axs[1, 1].imshow(data[4], cmap='gray')
    axs[1, 2].imshow(data[5], cmap='gray')
    axs[2, 0].imshow(data[6], cmap='gray')
    axs[2, 1].imshow(data[7], cmap='gray')
    axs[2, 2].imshow(data[8], cmap='gray')
    axs[3, 0].imshow(data[9], cmap='gray')
    axs[3, 1].imshow(data[10], cmap='gray')
    #axs[2, 3].imshow(data[11], cmap='gray')

    axs[0, 0].set_title('sigma=1.0 - sigma=1.4142')
    axs[0, 1].set_title('sigma=1.4142 - sigma=2.0')
    axs[0, 2].set_title('sigma=2.0 - sigma=2.8284')
    axs[1, 0].set_title('sigma=2.8284 - sigma=4.0')
    axs[1, 1].set_title('sigma=4.0 - sigma=5.6568')
    axs[1, 2].set_title('sigma=5.6568 - sigma=8.0')
    axs[2, 0].set_title('sigma=8.0 - sigma=11.3137')
    axs[2, 1].set_title('sigma=11.3137 - sigma=16.0')
    axs[2, 2].set_title('sigma=16.0 - sigma=22.6274')
    axs[3, 0].set_title('sigma=22.6274 - sigma=32.0')
    axs[3, 1].set_title('sigma=32.0 - sigma=45.2548')
    
    plt.show()

######################################################################
#
#
#  PART 1A
#
#
######################################################################

print_line()
print('\nPart A (5 points)')

'''
[2 points]
Download the input image file Assignment_MV_01_image_1.jpg from Canvas.  
Load the file and convert it into a single channel grey value image.

[1 point]
Make sure the data type is float32 to avoid any rounding errors. 
'''
fname = 'Assignment_MV_01_image_1.jpg'
img = cv2.imread(fname).astype(np.float32)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(20,10))
plt.imshow(gray, cmap='gray')
plt.title('Task 1A - Input image - Gray', fontsize=16)
plt.show()


'''
Verify that the image indeed contains floats.
'''
print()
print(gray[0])


'''
[2 points]
Determine the size of the image and resize the image to double its size.
'''
print('\nThe image dimensions are:', gray.shape)

x, y = gray.shape[0], gray.shape[1]
print('\nx: {}, y: {}'.format(x, y))

gray_resized = cv2.resize(gray, (y*2, x*2), interpolation=cv2.INTER_LINEAR)

print('The resized image dimensions are:', gray_resized.shape)

plt.figure(figsize=(20,10))
plt.imshow(gray_resized, cmap='gray')
plt.title('Task 1A - Resized input image - Gray', fontsize=16)
plt.show()


######################################################################
#
#
#  PART 1B
#
#
######################################################################

print_line()
print('\nPart B (6 points)')


'''
[4 points]
Create twelve Gaussian smoothing kernels with increasing 𝜎 = 2^𝑘⁄2, 𝑘 = 0, … ,11, and plot each of these kernels as image.  
Make sure that the window size is large enough to sufficiently capture the characteristic of the Gaussian. 
'''
num_gaussian_kernels = 12
sigmas = [ 2**(k/2) for k in range(0, num_gaussian_kernels) ]
print('\nSigma values are:', sigmas)

'''
Create Gaussian kernels.
The kernels will be of size (2*3*sgima + 1)x(2*3*sgima + 1), hence kernels will grow in size as sigma increases.
'''
def gaussian_filter(sigma):
  size = 2*np.ceil(3*sigma)+1
  x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
  kernel = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
  return kernel/kernel.sum()

gaussian_kernels = [ gaussian_filter(sigma) for sigma in sigmas ]

plot_grid34(gaussian_kernels, 'Task 1B - Gaussian Kernels for various sigma values')


'''
[2 points]
Apply these kernels to the resized input image from subtask A to create a scale-space representation and display all resulting scale-space images.
'''
scale_space = [ cv2.filter2D(gray_resized, -1, kernel) for kernel in gaussian_kernels ]
plot_grid34(scale_space, 'Task 1B - Scale space: Image blurred with various sigma values')


######################################################################
#
#
#  PART 1C
#
#
######################################################################

print_line()
print('\nPart C (3 points)')

'''
[3 points]**  
Use the scale-space representation from subtask B to calculate difference of Gaussian images at all scales. Display all resulting DoG images.
'''
'''
Create the DoG images by subtracting scale space images from each other.
According to our lecture slides we subtract k+1 from k to to derive a DoG image.
'''
dogs = [ scale_space[i] - scale_space[i+1] for i in range(0, num_gaussian_kernels-1) ]
plot_grid_dogs(dogs, 'Task 1C - Difference of Gaussians in scale space')


######################################################################
#
#
#  PART 1D
#
#
######################################################################

print_line()
print('\nPart D (3 points)')

'''
[3 points]
Find key-points by thresholding all DoGs from subtask C using a threshold of 𝑇 = 10.  
Suppress non-maxima in scale-space by making sure that the key-points have no neighbours, both in space as well as in scale, with higher values.  
The resulting key-points should comprise three coordinates (𝑥, 𝑦, 𝜎), two spatial and the scale at which they were detected.
'''
key_points = {}

def non_max_suppression(dogs, i, x, y, T, suppress_space):
    '''
    This function implements thresholding and gets the maximum intensity in the neighborhood of a given point (x, y).
    Returns False if larger intensity than (x, y, i) is found, True otherwise.
    '''
    # Get current value (x, y, i)
    cur_val = dogs[i][x][y]
    
    # Only continue if current value is greater or equal to threshold T
    if cur_val >= T:
        # Get the max value within the 3d cube
        cube_max = max([ np.max(dog[x-1:x+2, y-1:y+2]) for dog in suppress_space ])
        
        # Check whether or not the current value is greater or equal than cube_max
        #
        # Note: As current value is part of cube_max check, my if operator is '>='
        #       There is a slight risk a neighboring pixel in scale space has the same value.
        #       I chose this a being acceptable.
        #
        if cur_val >= cube_max:
            return True
        
    return False


'''
Iterate over relevant DoG slices. 

>> enumerate(sigmas[1:-2])
[1:-2] excludes the first and last DoG fom being used during non-max-suppression. Hence, these slices can never produce a key-point.

>> suppress_space = [dogs[i-1], dogs[i], dogs[i+1]]
The suppress_space for non_max_suppression always contains the current DoG and the 2 neighboring DoGs

>> range(1, dog.shape[0]) and range(1, dog.shape[1])
Here I ensure that in each DoG the outermost pixel are not used for key-point check, but they are part of the 26-neighborhood.


Perform non_max_suppression check for each remaining pixel.
If non_max_suppression returns True, add pixel (x, y, sigma) to key_points.
'''

# Set the threshold
T = 10

print('')
for (i, sigma) in enumerate(sigmas[1:-2]):
    i += 1
    print('DoG: {}, associated sigma: {}'.format(i, sigma))
    dog = dogs[i]
    suppress_space = [dogs[i-1], dogs[i], dogs[i+1]]
    
    for x in range(1, dog.shape[0]):
        for y in range(1, dog.shape[1]):
            if non_max_suppression(dogs, i, x, y, T, suppress_space):
                key_points[(x, y, sigma)] = {}
                key_points[(x, y, sigma)]['value'] = dog[x][y]

print('\nThesholding and non-max-suppression resulted in {} key-points.'.format(len(key_points)))


######################################################################
#
#
#  PART 1E
#
#
######################################################################

print_line()
print('\nPart E (4 points)')

'''
[4 points]
Calculate derivatives of all scale-space images from subtask B using the kernels 𝑑𝑥 = (1 0 −1) and 𝑑𝑦 = (1 0 −1)𝑇. Display the resulting derivative images at all scales.
'''
'''
Define kernels dx and dy as per definition.
'''
dx = np.array([[1, 0, -1]])
dy = np.array([[1, 0, -1]]).T

'''
Apply kernels dx and dy to any scale space image resulting in list of
dx- and dy-derivatives of scale space images.
'''
scale_space_dx = [ cv2.filter2D(ss_img, -1, dx) for ss_img in scale_space ]
scale_space_dy = [ cv2.filter2D(ss_img, -1, dy) for ss_img in scale_space ]

'''
Plot the derivative images after applying dx kernel.
'''
plot_grid34(scale_space_dx, 'Task 1E - dx gradient images in scale space')

'''
Plot the derivative images after applying dy kernel.
'''
plot_grid34(scale_space_dy, 'Task 1E - dy gradients in scale space')



######################################################################
#
#
#  PART 1F
#
#
######################################################################

print_line()
print('\nPart F (9 points)')

'''
[4 points]
Calculate the gradient length 𝑚_𝑞𝑟 and gradient direction 𝜃_𝑞𝑟 for the 7 × 7 grid of points (𝑞, 𝑟) ∈ {𝑥 + 3/2 𝑘𝜎 | 𝑘 = −3, . . ,3} × {𝑦 + 3/2 𝑘𝜎 | 𝑘 = −3, . . ,3}
sampled around each key-point (𝑥, 𝑦) and using the appropriate scale 𝜎 determined in subtask D and the correct gradient images from subtask E.

[1 point]
Also calculate a Gaussian weighting function w_qr = ... for each of the grid points.

[3 points]
Now create a 36-bin orientation histogram vector ℎ and accumulate the weighted gradient lengths 𝑤_𝑞𝑟*𝑚_𝑞𝑟 for each grid point (𝑞, 𝑟) where the gradient direction 𝜃𝑞𝑟 falls into this particular bin.

[1 point]
Use the maximum of this orientation histogram to determine the orientation of the key-point.
'''

# https://aishack.in/tutorials/sift-scale-invariant-feature-transform-keypoint-orientation/
# https://stackoverflow.com/questions/1707151/finding-angles-0-360
# https://www.rapidtables.com/math/trigonometry/arctan.html#definition

'''
Task 1F-1
    Create a grid (here: dictionary) representing sampled coordinate around key-point (x, y, sigma).

    Example for (36, 413, 1)
        ...
        (0, 0): (36, 413),    <-- key-point
        (0, 1): (36, 414), 
        (0, 2): (36, 416),
        (0, 3): (36, 417),
        ...
'''
k = [ -3, -2, -1, 0, 1, 2, 3 ]

def get_grid(k, x, y, sigma):
    grid = {}
    for i in k:
        for j in k:
            x2 = int(x + 3/2 * i * sigma)
            y2 = int(y + 3/2 * j * sigma)
            grid[(i, j)] = x2, y2
            
            '''
            Check if any grid point lies outside the image dimensions and return None
            '''
            if x2 < 0 or y2 < 0 or x2 > 1535 or y2 > 2047:
                return None

    return grid


'''
Task 1F-2
    Also calculate a Gaussian weighting function w_qr = ... for each of the grid points.

    Example for (36, 413, 1)
    [[0.00129557 0.00393558 0.00766547 0.00957301 0.00766547 0.00393558 0.00129557]
     [0.00393558 0.01195525 0.02328564 0.02908025 0.02328564 0.01195525 0.00393558]
     [0.00766547 0.02328564 0.04535423 0.05664058 0.04535423 0.02328564 0.00766547]
     [0.00957301 0.02908025 0.05664058 0.07073553 0.05664058 0.02908025 0.00957301]  <-- 0.07073553 is wqr value for key-point
     [0.00766547 0.02328564 0.04535423 0.05664058 0.04535423 0.02328564 0.00766547]
     [0.00393558 0.01195525 0.02328564 0.02908025 0.02328564 0.01195525 0.00393558]
     [0.00129557 0.00393558 0.00766547 0.00957301 0.00766547 0.00393558 0.00129557]]
'''
def weight_qr(sigma):
    wqr = np.zeros((7,7))
    for q in range(-3, 4):
        for r in range(-3, 4):
            wqr[q+3][r+3] = np.exp(-(q**2 + r**2)/(9*sigma**2/2))/(9*np.pi*sigma**2/2)
    return wqr

'''
Iterate over all key-points
'''
invalid_kp = []

for kp, v in key_points.items():
    '''
    Task 1F-1
    '''
    # Extract relevant information:
    #   - x, y, sigma
    #   - scale level
    #   - (q,r) grid
    x, y, sigma = kp[0], kp[1], kp[2]
    scale = sigmas.index(sigma)
    grid = get_grid(k, x, y, sigma)
    
    '''
    Only continue if grid generation created valid pixel locations.
    Else add current key_point to list of invalid key_points and continue with next key_point.
    '''
    if not grid:
        invalid_kp.append(kp)
        continue
    
    '''
    Task 1F-2
    Task 1F-3
    '''
    wqr = weight_qr(sigma)
    hist = np.zeros([36])
    
    '''
    For each key-point calculate
      - theta (use arctan2 function and *180/pi to get range from [0..359])
      - mqr 
      - wqr*mqr
      - orientation histogram
    '''
    for (q, r), (x_sample, y_sample) in grid.items():
        dx = scale_space_dx[scale][x_sample][y_sample]
        dy = scale_space_dy[scale][x_sample][y_sample]
        
        '''
        Task 1F-1
        Task 1F-2
        '''
        theta = np.arctan2(dy, dx)*180/np.pi
        mqr = np.sqrt(dx**2 + dy**2)
        wqr_mqr = wqr[q+3][r+3] * mqr

        '''
        Task 1F-3
            Now create a 36-bin orientation histogram vector ℎ and accumulate...
        '''
        hist[int(theta // 10)] +=  wqr_mqr

    
    '''
    Task 1F-4
        Use the maximum of this orientation histogram to determine the orientation of the key-point.
        Add "+5" to account for bin width, i.e. 10/2 = 5
    '''
    max_orientation = hist.argmax() * 10 + 5
    v['orientation'] = max_orientation


'''
Delete key_points with invalid grid entries.
'''
for kp in invalid_kp:
    del key_points[kp]

print('\nNumber of key-points left after grid generation: {}'.format(len(key_points)))


######################################################################
#
#
#  PART 1G
#
#
######################################################################

print_line()
print('\nPart G (3 points)')

'''
[3 points]
Draw all the key-points into the input image using a circle with 3𝜎 radius and a line from the key-point centre to the circle radius to indicate the orientation
(see example for a single key-point on the right).
Display the resulting output image with all the key-points.
'''
plt.figure(figsize=(30,15))

final = cv2.imread(fname)
final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

'''
For each key-point...
'''
for k, v in key_points.items():
    '''
    Get x, y, and sigma information
    Draw circle around x,y with radius 3*sigma
    '''
    x, y, sigma = int(k[0]/2), int(k[1]/2), k[2]
    cv2.circle(final, (y, x), int(3*sigma), (0,255,0), 1)
    
    '''
    Calculate the end pixel with the direction line
    '''
    theta = v['orientation']/180*np.pi
    x2 = x - int(3*sigma * math.sin(theta))
    y2 = y + int(3*sigma * math.cos(theta))

    cv2.line(final, (y, x), (y2, x2), (255,0,0), 2)

plt.imshow(final/np.max(final))
plt.title('Task 1G - Input image with key points and orientation (full) - Color', fontsize=16)
plt.show()


'''
To improve the visual output only show 300 key points
'''

plt.figure(figsize=(30,15))

final = cv2.imread(fname)
final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

limit = 300
kp_list = random.sample(list(key_points.keys()), limit)

for kp in kp_list:
    x, y, sigma = int(kp[0]/2), int(kp[1]/2), kp[2]
    cv2.circle(final, (y, x), int(3*sigma), (0,255,0), 1)

    '''
    Calculate the end pixel with the direction line
    '''
    theta = key_points[kp]['orientation']/180*np.pi
    x2 = x - int(3*sigma * math.sin(theta))
    y2 = y + int(3*sigma * math.cos(theta))

    cv2.line(final, (y, x), (y2, x2), (255,0,0), 2)

plt.imshow(final/np.max(final))
plt.title('Task 1G - Input image with key points and orientation (sampled) - Color', fontsize=16)
plt.show()


print_line()
