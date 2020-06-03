#!/usr/bin/env python

import cv2
import numpy as np
import random
import os
import glob
import pprint


pp = pprint.PrettyPrinter(indent=4)


################################################################
#
# TASK 1
#
################################################################

'''
1 A 

[3 points]
    Download the file Assignment_MV_02_calibration.zip from Canvas and load all calibration images contained in this archive. 
    Extract and display the checkerboard corners to subpixel accuracy in all images using the OpenCV calibration tools [3 points].

'''

# Defining the dimensions of checkerboard
CHECKERBOARD = (5,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./Assignment_MV_02_calibration/*.png')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Rescale the image to allow corner detection in Assignment_MV_02_calibration_5.png
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation = cv2.INTER_AREA)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    Refine checkers corners to subpixel accuracy
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        # Rescale pixel values to match original frame size.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) * (100 / scale_percent)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


'''
1 B 

[1 point]
    Determine and output the camera calibration matrix 𝑲 using the OpenCV calibration tools [1 point].

'''

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\nCamera matrix :")
print(K)


'''
1 C

[2 points]
    Download the file Assignment_MV_02_video.mp4 from Canvas and open it for processing. 
    Identify good features to track in the first frame [1 point] using the OpenCV feature extraction and tracking functions. 
    Refine the feature point coordinates to sub-pixel accuracy [1 point].


1 D

[2 points]
    Use the OpenCV implementation of the KLT algorithm to track these features across the whole image sequence [1 point]. 
    Make sure to refine the feature point coordinates to sub-pixel accuracy [1 point] in each step.
'''

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_tracks(filename):
    camera = cv2.VideoCapture(filename)

    # initialise features to track
    while camera.isOpened():
        ret,img= camera.read()
        if ret:
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #p0 = cv2.goodFeaturesToTrack(new_img, mask = None, **feature_params)
            p0 = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7) 
            p0 = cv2.cornerSubPix(new_img, p0, (11,11), (-1,-1), criteria)
            break

    # initialise tracks
    index = np.arange(len(p0))
    tracks = {}
    for i in range(len(p0)):
        tracks[index[i]] = {0:p0[i]}
                
    frame = 0
    while camera.isOpened():
        ret,img= camera.read()                 
        if not ret:
            break

        frame += 1

        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # calculate optical flow
        if len(p0)>0:
            #p1, st, err  = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None, **lk_params)
            p1, st, err  = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None) 
            p1 = cv2.cornerSubPix(new_img, p1, (11,11), (-1,-1), criteria)
            
            # visualise points
            for i in range(len(st)):
                if st[i]:
                    cv2.circle(img, (p1[i,0,0],p1[i,0,1]), 2, (0,0,255), 2)
                    cv2.line(img, (p0[i,0,0],p0[i,0,1]), (int(p0[i,0,0]+(p1[i][0,0]-p0[i,0,0])*5),int(p0[i,0,1]+(p1[i][0,1]-p0[i,0,1])*5)), (0,0,255), 2)
            
            p0 = p1[st==1].reshape(-1,1,2)
            index = index[st.flatten()==1]

        # update tracks
        for i in range(len(p0)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0[i]
            else:
                tracks[index[i]] = {frame: p0[i]}

        # visualise last frames of active tracks
        for i in range(len(index)):
            for f in range(frame-20,frame):
                if (f in tracks[index[i]]) and (f+1 in tracks[index[i]]):
                    cv2.line(img,
                             (tracks[index[i]][f][0,0],tracks[index[i]][f][0,1]),
                             (tracks[index[i]][f+1][0,0],tracks[index[i]][f+1][0,1]), 
                             (0,255,0), 1)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        
        cv2.imshow("camera", img)
        
    camera.release()
        
    return tracks, frame


# Get the tracks for all features
tracks, frames = get_tracks("Assignment_MV_02_video.mp4")

print('\nThe movies contains {} frames'.format(frames))
print('Total features found: {}'.format(len(tracks.keys())))


################################################################
#
# TASK 2
#
################################################################

'''
2 A 

[2 points]
    Extract and visualise the feature tracks calculated in task 1 which 
    are visible in both the first and the last frame to establish 
    correspondences 𝒙𝒊↔𝒙𝒊′ between the two images [2 points]. 
    Use Euclidean normalised homogeneous vectors.

'''

def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame=0
    while camera.isOpened():
        ret,img= camera.read()        
        if not ret:
            break
        if frame in frames:
            result[frame] = img        
        frame += 1
        if frame>last_frame:
            break

    return result

def get_correspondences(tracks, frame1, frame2):
    correspondences = []
    for track in tracks:
        if (frame1 in tracks[track]) and (frame2 in tracks[track]):
            x1 = [tracks[track][frame1][0,0],tracks[track][frame1][0,1],1]
            x2 = [tracks[track][frame2][0,0],tracks[track][frame2][0,1],1]
            correspondences.append((track, np.array(x1), np.array(x2)))
    return correspondences

def display_tracks(images, f1, f2, correspondences, tracks, track_ids):

    # Create a mask image for drawing purposes
    mask = np.zeros_like(images[f2])

    # Add final points to mask
    for corr in correspondences:
        cv2.circle(mask, (int(corr[2][0]), int(corr[2][1])), 2, (0,0,255), 2)

    # Add historic points to mask
    for track_id in track_ids:
        for frame, point in tracks[track_id].items():
            cv2.circle(mask, (int(point[0][0]), int(point[0][1])), 1, (0,255,0), 1)

    output = cv2.add(images[f2], mask)

    # Display mask and image
    cv2.imshow('Prepared Mask', mask)
    cv2.imshow('Output image', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output, mask


# Define first and last frame numbers
f1, f2 = 0, frames

# Get first and last frame
images = extract_frames("Assignment_MV_02_video.mp4", [f1,f2])

# Get the features that exist in both frames
correspondences = get_correspondences(tracks, f1, f2)
track_ids = sorted([corr[0] for corr in correspondences])

print('\nTracks found in both frames:', len(track_ids), '\n', track_ids, '\n')

output, mask = display_tracks(images, f1, f2, correspondences, tracks, track_ids)


'''
2 B 

[2 points]
    Calculate the mean feature coordinates 𝝁=1𝑁Σ𝒙𝒊𝑖 and 𝝁′=1𝑁Σ𝒙𝒊′𝑖 in the first and the last frame [2 points].

[2 points]
    Also calculate the corresponding standard deviations 𝜎=√1𝑁Σ(𝒙𝒊−𝝁)2𝑖 and 𝜎′=√1𝑁Σ(𝒙𝒊′ −𝝁′)2𝑖 (where ( )2 denotes the element-wise square) [2 points].

[2 points]
    Normalise all feature coordinates and work with 𝒚𝒊=𝑻𝒙𝒊 and 𝒚𝒊′ =𝑻′𝒙𝒊′ which are translated and scaled using the homographies [2 points].
'''

def calculate_stats(correspondences):
    n = len(correspondences)
    
    # Get points
    x1 = [ p1[0] for frame, p1, p2 in correspondences ] 
    y1 = [ p1[1] for frame, p1, p2 in correspondences ]
    x2 = [ p2[0] for frame, p1, p2 in correspondences ]
    y2 = [ p2[1] for frame, p1, p2 in correspondences ]

    # Calculate mu's
    mu1 = [ sum(x1) / n, sum(y1) / n]
    mu2 = [ sum(x2) / n, sum(y2) / n]

    # Calculate sigma's
    sigma1 = [ np.sqrt(sum(np.square(x1 - mu1[0])) / n), np.sqrt(sum(np.square(y1 - mu1[1])) / n) ]
    sigma2 = [ np.sqrt(sum(np.square(x2 - mu2[0])) / n), np.sqrt(sum(np.square(y2 - mu2[1])) / n) ]

    # Print output
    print('\nMean  feat coord  frame0: {}'.format(mu1))
    print('Stdev feat coord  frame0: {}'.format(sigma1))
    print('Mean  feat coord frame30: {}'.format(mu2))
    print('Stdev feat coord frame30: {}\n'.format(sigma2))

    return mu1, sigma1, mu2, sigma2

def get_norm_vectors():
    T1 = np.array([[1/sigma1[0], 0, -mu1[0]/sigma1[0]],
                   [0, 1/sigma1[1], -mu1[1]/sigma1[1]],
                   [0, 0, 1]])

    T2 = np.array([[1/sigma2[0], 0, -mu2[0]/sigma2[0]],
                   [0, 1/sigma2[1], -mu2[1]/sigma2[1]],
                   [0, 0, 1]])

    return T1, T2

def normalise_coordinates(correspondences, mu1, sigma1, mu2, sigma2):
    T1, T2 = get_norm_vectors()

    normalised_correspondences = []
    for track, p1, p2 in correspondences:
        p1n = np.matmul(T1, p1)
        p2n = np.matmul(T2, p2)
        normalised_correspondences.append((track, p1n, p2n))
    
    return normalised_correspondences


def display_metrics_image(mask, images, f1, f2, mu1, sigma1, mu2, sigma2):
    # Add mean and stdev of last frame to mask
    cv2.circle(mask, (int(mu2[0]), int(mu2[1])), 5, (255,0,0), 3)
    cv2.line(mask, (int(mu2[0])-int(sigma2[0]), int(mu2[1])), (int(mu2[0])+int(sigma2[0]), int(mu2[1])), (255,0,0), 2)
    cv2.line(mask, (int(mu2[0]), int(mu2[1])-int(sigma2[1])), (int(mu2[0]), int(mu2[1])+int(sigma2[1])), (255,0,0), 2)
    output = cv2.add(images[f2], mask)

    # Display mask and image
    cv2.imshow('Prepared Mask with metrics', mask)
    cv2.imshow('Output image with metrics', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

mu1, sigma1, mu2, sigma2 = calculate_stats(correspondences)
normalised_correspondences = normalise_coordinates(correspondences, mu1, sigma1, mu2, sigma2)
display_metrics_image(mask, images, f1, f2, mu1, sigma1, mu2, sigma2)


'''
2 C

    (a) [1 point]
        Select eight feature correspondences at random [1 point] 
        
    (b) [1 point]
        and build a matrix comprising the eight corresponding rows 𝒂𝒊𝑻=𝒚𝒊𝑻⊗𝒚𝒊′ to calculate 
        the fundamental matrix using the 8-point DLT algorithm [1 point].


2 D

    (a) [1 point]
        Use the 8-point DLT algorithm to calculate the fundamental matrix 𝑭̂ for the eight selected normalised correspondences 𝒚𝒊↔𝒚𝒊′ [1 point]. 

    (b) [1 point]
        Make sure that 𝑭̂ is singular [1 point]. 
        
    (c) [1 point]    
        Apply the normalisation homographies to 𝑭̂ to obtain the fundamental matrix 𝑭=𝑻′𝑻𝑭̂ 𝑻 [1 point].

'''


def calculate_fundamental_matrix(track_ids, normalised_correspondences):
    T1, T2 = get_norm_vectors()

    # Task 2 C (a) 
    samples_dlt = list(random.sample(range(len(normalised_correspondences)),8))
    samples_out = set(range(len(normalised_correspondences))).difference(samples_dlt)
    
    # Task 2 C (b) 
    A = np.zeros((0,9))
    for i in samples_dlt:
        track, y1, y2 = normalised_correspondences[i]
        Ai = np.kron(y1.T, y2.T)
        A = np.append(A,[Ai],axis=0)

    # Task 2 D (a) 
    U,S,V = np.linalg.svd(A)    
    F_hat = V[8,:].reshape(3,3).T

    # Task 2 D (b) 
    U,S,V = np.linalg.svd(F_hat)
    F_hat = np.matmul(U,np.matmul(np.diag([S[0],S[1],0]),V))
    
    # Task 2 D (c) 
    F = np.matmul(T2.T, np.matmul(F_hat, T1))
        
    return F, samples_dlt, samples_out

F, F_samples, out_samples = calculate_fundamental_matrix(track_ids, normalised_correspondences)


'''
2 E

    (a) [1 point]
        For the remaining feature correspondences 𝒙𝒊↔𝒙𝒊′ not used in the 8-point algorithm calculate the value of the model equation [1 point].
        
    (b) [1 point]
        Also calculate the variance of the model equation [1 point].


2 F

    (a) [1 point]
        Determine for each of these correspondences if they are an outlier with respect to the selection of the 
        eight points or not by calculating the test statistic [1 point].

    (b) [1 point]
        Use an outlier threshold of 𝑇𝑖>6.635 [1 point]. Sum up the test statistics over all inliers [1 point].

'''

def find_outliers(F, F_samples, track_ids, correspondences):
    
    # Task 2 E (b)
    Cxx = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

    # Task 2 F (b)
    Ti_threshold = 6.635
    inlier_sum = 0
    inliers = []
    outliers = []

    # Add DLT Points to inliers
    #
    # Intuition: DLT points define F, so they must be inliers.
    #
    dlt_points = list(map(correspondences.__getitem__, F_samples))
    for track, p1, p2 in dlt_points:
        inliers.append([track, p1, p2])

    # Task 2 E (a)
    unused = [ idx for idx in range(len(track_ids)) if idx not in F_samples ]
    unused_samples = list(map(correspondences.__getitem__, unused))

    for track, p1, p2 in unused_samples:
        # Task 2 E (a)
        gi = np.matmul(p2.T, np.matmul(F, p1))

        # Task 2 E (b)
        sigma_i = np.matmul(p2.T, np.matmul(F, np.matmul(Cxx, np.matmul(F.T, p2)))) + \
                  np.matmul(p1.T, np.matmul(F.T, np.matmul(Cxx, np.matmul(F, p1))))

        # Task 2 F (a)
        Ti = np.square(gi) / sigma_i
        if Ti <= Ti_threshold:
            inlier_sum += Ti
            inliers.append([track, p1, p2])
        else:
            outliers.append([track, p1, p2])
    
    return inliers, outliers, inlier_sum


inliers, outliers, inlier_sum = find_outliers(F, F_samples, track_ids, correspondences)

print('\nNum of inliers : {}'.format(len(inliers)))
print('Num of outliers: {}'.format(len(outliers)))
print('Inliers sum    : {}\n'.format(inlier_sum))


'''
2 G

    (a) [1 point]
        Repeat the above procedure 10000 times for different random selections 
        of correspondences [1 point].
        
    (b) [1 point]
        Select the fundamental matrix and remove all outliers for the selection of 
        eight points which yielded the least number of outliers [1 point].

    (c) [1 point]
        Break ties by looking at the sum of the test statistic over the inliers [1 point].

'''

def F_loop(iterations, track_ids, correspondences, normalised_correspondences):

    BEST = []                       # sample: [F, F_samples, inliers, outliers, inlier_sum]
    NUM_OUTLIERS = len(tracks) + 1

    for i in range(1, iterations + 1):
        if i % 500 == 0:
            print('Iterations completed: {}'.format(i))

        # Calculate F, inliers, outliers and metric
        F, F_samples, out_samples = calculate_fundamental_matrix(track_ids, normalised_correspondences)
        inliers, outliers, inlier_sum = find_outliers(F, F_samples, track_ids, correspondences)

        # Check if new best F was found
        if len(outliers) == NUM_OUTLIERS:
            if  inlier_sum < BEST[4]:
                BEST = [F, F_samples, inliers, outliers, inlier_sum]
        elif len(outliers) < NUM_OUTLIERS:
            BEST = [F, F_samples, inliers, outliers, inlier_sum]
            NUM_OUTLIERS =  len(outliers)

    #print('\n', len(BEST[2]), len(BEST[3]), BEST[4])

    print('\nBest F:')
    print(BEST[0])

    print('\nInliers :', len(BEST[2]))
    print('Outliers:', len(BEST[3]))
    print('Inliers Sum:', BEST[4])

    return BEST


iterations = 10000
BEST = F_loop(iterations, track_ids, correspondences, normalised_correspondences)

'''
2 H

    (a) [1 point]
        Adapt the display of feature tracks implemented in subtask A to indicate 
        which of these tracks are inliers and which tracks are outliers [1 point]. 
        
    (b) [2 points]  
        Also calculate and output the coordinates of the two epipoles? [2 points]

'''

def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)    
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)    
    e2 = V[2,:]

    return e1, e2    

def calculate_epipolar_line(F, x, width, height):
    l = np.matmul(F, x)
    l1 = np.cross([0,0,1],[width-1,0,1])
    l2 = np.cross([0,0,1],[0,height-1,1])
    l3 = np.cross([width-1,0,1],[width-1,height-1,1])
    l4 = np.cross([0,height-1,1],[width-1,height-1,1])
    x1 = np.cross(l,l1)
    x2 = np.cross(l,l2)
    x3 = np.cross(l,l3)
    x4 = np.cross(l,l4)
    x1 /= x1[2]
    x2 /= x2[2]
    x3 /= x3[2]
    x4 /= x4[2]
    result = []
    if (x1[0]>=0) and (x1[0]<=width):
        result.append(x1)
    if (x2[1]>=0) and (x2[1]<=height):
        result.append(x2)
    if (x3[1]>=0) and (x3[1]<=height):
        result.append(x3)
    if (x4[0]>=0) and (x4[0]<=width):
        result.append(x4)
    return result[0],result[1]


def outlier_image(images, f1, f2, tracks, correspondences, BEST):

    F_samples   = BEST[1]
    inliers     = BEST[2]
    outliers    = BEST[3]

    # Create a mask image for drawing purposes
    mask1 = np.zeros_like(images[f1])
    mask2 = np.zeros_like(images[f2])

    for inlier in inliers:
        track_id = inlier[0]
        cv2.circle(mask2, (int(tracks[track_id][30][0][0]), int(tracks[track_id][30][0][1])), 2, (0,255,0), 2)

        for _, point in tracks[track_id].items():
            cv2.circle(mask2, (int(point[0][0]), int(point[0][1])), 1, (0,255,0), 1)

    for outlier in outliers:
        track_id = outlier[0]
        cv2.circle(mask2, (int(tracks[track_id][30][0][0]), int(tracks[track_id][30][0][1])), 2, (0,0,255), 2)
        for _, point in tracks[track_id].items():
            cv2.circle(mask2, (int(point[0][0]), int(point[0][1])), 1, (0,0,255), 1)


    width = images[30].shape[1]
    height = images[30].shape[0]
    F = BEST[0]

    e1,e2 = calculate_epipoles(F)
    print('\nEpipoles:')
    print(e1/e1[2])
    print(e2/e2[2])

    cv2.circle(mask1, (int(e1[0]/e1[2]),int(e1[1]/e1[2])), 3, (0,255,255), 2)
    cv2.circle(mask2, (int(e2[0]/e2[2]),int(e2[1]/e2[2])), 3, (0,255,255), 2)

    x = np.array([0.5*width, 
                  0.5*height,
                  1])
    x1, x2 = calculate_epipolar_line(F, x, width, height)

    cv2.circle(mask1, (int(x[0]/x[2]),int(x[1]/x[2])), 3, (0,0,255), 2)
    cv2.line(mask2, (int(x1[0]/x1[2]),int(x1[1]/x1[2])), (int(x2[0]/x2[2]),int(x2[1]/x2[2])), (0,255,255), 2)


    output1 = cv2.add(images[f1], mask1)
    output2 = cv2.add(images[f2], mask2)

    # Display mask and image
    cv2.imshow('Outlier Mask', mask2)
    cv2.imshow('Outlier image 1 with epipoles', output1)
    cv2.imshow('Outlier image 2 with epipolar line', output2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return e1, e2

e1, e2 = outlier_image(images, f1, f2, tracks, correspondences, BEST)



################################################################
#
# TASK 3
#
################################################################

'''
3 A 
    (a) [1 point]
        Use the fundamental matrix 𝑭 determined in task 2 and the calibration matrix 𝑲 determined in task 1 to calculate the essential matrix 𝑬 [1 point].

    (b) [1 point]
        Make sure that the non-zero singular values of 𝑬 are identical [1 point].

    (c) [1 point]
        Also make sure that the rotation matrices of the singular value decomposition have positive determinants [1 point].

'''

def get_essential_matrix(K, F):
    # 3A-a
    # Calculate the E estimate from K and F
    E_hat = np.matmul(K.T, np.matmul(F, K))
    print('\nEssential Matrix E_hat: \n', E_hat)

    # 3A-b
    # Get singluar value decomposition
    # Get lambda
    U,S,V = np.linalg.svd(E_hat)

    lambd = (S[0] + S[1])/2

    # 3A-c
    # Get singluar value decomposition
    if np.linalg.det(U) < 0: 
        U[:, 2] *= -1
    if np.linalg.det(V) < 0: 
        V[2, :] *= -1

    # Update E_hat with updated U,S,V
    # Ensure non-zero singular values of 𝑬 are identical
    E = np.matmul(U, np.matmul(np.diag([lambd, lambd, 0]), V.T))
    print('\nEssential Matrix E with equal lambda: \n', E)

    return E, U, V, lambd

E, U, V, lambd = get_essential_matrix(K, F)

'''
3 B
    (a) [3 points]
        Determine the four potential combinations of rotation matrices 𝑹 and translation vector 𝒕 between the first and the last frame [3 points]. 

    (b) [1 point]
        Assume the camera was moving at 50km/h and that the video was taken at 30fps to determine the scale of the baseline 𝒕 in meters [1 point].

'''

def get_R_t(E, U, V, lambd):
    # Task 3 B (b)
    #
    # Because our sequence is 30 frames and 1 sec has 30 frames,
    # we can simplify the calculation of beta.
    #
    beta = (50000/3600)

    W = np.array([  [0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])

    Z = np.array([  [ 0, 1, 0],
                    [-1, 0, 0],
                    [ 0, 0, 0]])

    # Task 3 B (a)
    #
    # Note: r1 and r2 are transposed, as the lecture slide shows formula for R.T
    #
    s1 = -beta * np.matmul(U, np.matmul(Z, U.T))
    s2 =  beta * np.matmul(U, np.matmul(Z, U.T))
    r1 = np.matmul(U, np.matmul(W, V.T)).T
    r2 = np.matmul(U, np.matmul(W.T, V.T)).T

    #
    # According to Zisserman/Hartley t = -u_3 and +u_3
    # with u_3 being the 3rd column of U
    #
    t1 = -beta * U[:, 2]
    t2 =  beta * U[:, 2]

    '''r1t1 = np.matmul(np.linalg.inv(r1), s1)
    r1t2 = np.matmul(np.linalg.inv(r1), s2)
    r2t1 = np.matmul(np.linalg.inv(r2), s1)
    r2t2 = np.matmul(np.linalg.inv(r2), s2)
    print(r1t1)
    print(r1t2)
    print(r2t1)
    print(r2t2)
    '''


    #
    # Debug prints and validations
    #
    print('\nVerify that R1 and R2 are rotation matrices with: Det(R) = 1  and  R.TxR = I ')
    print('Det(R1):', np.linalg.det(r1))
    print('Det(R2):', np.linalg.det(r2))
    print('R1:\n', np.matmul(r1.T, r1).round())
    print('R2:\n', np.matmul(r2.T, r2).round())

    print('\nt1:', t1)
    print('\nt2:', t2)

    print('\nS1:\n', s1)
    print('\nS2:\n', s2)
    
    #
    # Prepare the 4 solutions
    #
    RT = {
        0: { 'sol': (r1, t1), 'inliers': [], 'outliers': [] },
        1: { 'sol': (r1, t2), 'inliers': [], 'outliers': [] },
        2: { 'sol': (r2, t1), 'inliers': [], 'outliers': [] },
        3: { 'sol': (r2, t2), 'inliers': [], 'outliers': [] },
    }

    return RT


RT = get_R_t(E, U, V, lambd)


'''
3 C
    (a) [1 point]
        Calculate for each inlier feature correspondence determined in task 2 and each potential solution calculated 
        in the previous subtask the directions 𝒎 and 𝒎′ of the 3d lines originating from the centre of projection towards the 3d points [1 point]
            𝑿[𝜆] = 𝜆𝒎
        and 𝑿[𝜇] = 𝒕+𝜇𝑹𝒎′

    (b) [1 point]
        Then calculate the unknown distances 𝜆 and 𝜇 by solving the linear equation system [1 point]
        (𝒎𝑻𝒎−𝒎𝑻𝑹𝒎′𝒎𝑻𝑹𝒎′−𝒎′𝑻𝒎′)(𝜆𝜇) = (𝒕𝑻𝒎𝒕𝑻𝑹𝒎′)

    (c) [1 point]
        to obtain the 3d coordinates of the scene points [1 point]. 
        
    (d) [1 point]
        Determine which of the four solutions calculated in the previous subtask is correct by selecting the one 
        where most of the scene points are in front of both frames, i.e. where both distances 𝜆>0 and 𝜇>0 [1 point]. 
        
    (e) [1 point]    
        Discard all points, which are behind either of the frames for this solution as outliers [1 point].
'''

def calculate_3d_scene(BEST, RT):
    inliers = BEST[2]

    for track, p1, p2 in inliers:

        for k, v in RT.items():
            r = v['sol'][0]
            t = v['sol'][1]

            # Task 3 C (a)
            m1 = np.matmul(np.linalg.inv(K), p1)
            m2 = np.matmul(np.linalg.inv(K), p2)

            # Task 3 C (b)
            a = np.array([
                [np.matmul(m1.T, m1), -np.matmul(m1.T, np.matmul(r, m2))],
                [np.matmul(m1.T, np.matmul(r, m2)), -np.matmul(m2.T, m2)]])
            b = np.array([np.matmul(t.T, m1), np.matmul(t.T, np.matmul(r, m2))])

            x = np.linalg.solve(a, b)

            lambd = x[0]
            mu    = x[1]

            # Task 3 C (c)
            # 3D Scene coordinates: X1, X2
            X1 = lambd * m1
            X2 = t + mu * np.matmul(r, m2)

            X3 = (X1 + X2)/2


            # Task 3 C (d)
            # Task 3 C (e)
            if lambd > 0 and mu > 0:
                v['inliers'].append((track, p1, p2, X1, X2, X3))
            else:
                v['outliers'].append((track, p1, p2, X1, X2, X3))
    
    return RT

RT = calculate_3d_scene(BEST, RT)

# Task 3 C (e)
print('\nInliers  per solution:', [ len(v['inliers']) for k, v in RT.items() ])
print('Outliers per solution:', [ len(v['outliers']) for k, v in RT.items() ])
print('Best         solution:', np.argmax([ len(v['inliers']) for k, v in RT.items() ]))

best_solution = RT[np.argmax([ len(v['inliers']) for k, v in RT.items() ])]



'''
3 D
    (a) [2 points]
        Create a 3d plot to show the two camera centres and all 3d points [2 points].
'''   

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def plot_3d(best_solution):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # X3's - i.e. the average between X1 and X2
    xs_in3 = [ p[5][0] for p in best_solution['inliers'] ]
    ys_in3 = [ p[5][1] for p in best_solution['inliers'] ]
    zs_in3 = [ p[5][2] for p in best_solution['inliers'] ]

    # Camera center frame0
    C0 = [0, 0, 0]

    # Camera center frame30
    t = best_solution['sol'][1]
    C2 = t

    ax.scatter(C0[0], C0[1], C0[2], c='red', label='cam center f0')
    ax.scatter(C2[0], C2[1], C2[2], c='blue', label='cam center f30')
    ax.scatter(xs_in3, ys_in3, zs_in3, c='green', label='Inliers 3D Scene')
    ax.legend()

    plt.show()

plot_3d(best_solution)


'''
3 E
    (a) [2 points]
        Project the 3d points into the first and the last frame [2 points] 
        
    (b) [2 points]
        and display their position in relation to the corresponding features to visualise the reprojection error [2 points].
'''  

def show_reconstructed_points(best_solution):

    # Create a mask image for drawing purposes
    mask1 = np.zeros_like(images[f2])
    mask2 = np.zeros_like(images[f2])


    for inlier in best_solution['inliers']:
        new_x = np.matmul(K, inlier[5])
        new_x = new_x / new_x[2]

        cv2.circle(mask1, (int(new_x[0]), int(new_x[1])), 2, (0,255,0), 2)
        cv2.line(mask1, (int(new_x[0]), int(new_x[1])), (int(inlier[1][0]), int(inlier[1][1])), (0,0,255), 2)

        cv2.circle(mask2, (int(new_x[0]), int(new_x[1])), 2, (0,255,0), 2)
        cv2.line(mask2, (int(new_x[0]), int(new_x[1])), (int(inlier[2][0]), int(inlier[2][1])), (0,0,255), 2)

    output1 = cv2.add(images[f1], mask1)
    output2 = cv2.add(images[f2], mask2)

    # Display mask and image
    cv2.imshow('Frame0 with projection error', output1)
    cv2.imshow('Frame30 with projection error', output2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

show_reconstructed_points(best_solution)