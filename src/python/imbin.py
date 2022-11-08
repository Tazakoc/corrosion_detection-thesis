#!/usr/bin/env python3

"""imbin.py: Colour quantisation based on covariance eigen values

IMBIN utilises a binary tree approach to split nodes from max
of eigenvalues and store image resulting clusters.
Conditions for collecting the most significant tree node is also included.
"""

import numpy as np
from numpy import linalg as LA
import cv2 as cv
import matplotlib.pyplot as plt
import wx

__author__ = "G. Chliveros, and K. Papakosntantinou"
__copyright__ = "Copyright 2019, WatchOver Project"
__credits__ = ["", "", "", ""] # people who reported bug fixes
__license__ = "LGPL"
__version__ = "1.0.1"
__maintainer__ = "TBA"
__email__ = "TBA"
__status__ = "Prototype" # "Prototype", "Development", or "Production"


#=============================================================================#
# a dialogueBox to select the image filename from widget
#=============================================================================#
def get_path(wildcard):
    wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path
## end of dialogueBox

#=============================================================================#
#          Frame normalization function                                       #
#=============================================================================#
def frame_proc(img, height, width):
    '''
    Image processing function that initially converts input image to RGB,
    resizes the input image to the desired, user-defined dimentions by a
    bicubic interpolation over 4x4 pixel neighborhodd (INTER_CUBIC).
    Finally, the now resized image is normalized by means of MINMAX.

    The initial conversion to RGB occurs due to how OpenCv implements RGB
    colour codes (BGR to RGB).
    
    Parameters
    ----------
    img : int multidimensional array
        Input frame.
    height : int
        Frame height.
    width : int
        Frame width.
        
    Returns
    -------
    img_color : int64
        Normalized input frame.
        
    '''
    # Convert from BGR to RGB
    img_color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    #=============================================================================#
    #    -INTER_NEAREST  - a nearest-neighbor interpolation                       #
    #    -INTER_LINEAR   - a bilinear interpolation (used by default)             #
    #    -INTER_AREA     - resampling using pixel area relation. It may be a      #
    #                      preferred method for image decimation, as it gives     #
    #                      moire’-free results. But when the image is zoomed,     #
    #                      it is similar to the INTER_NEAREST method.             #
    #    -INTER_CUBIC    - a bicubic interpolation over 4x4 pixel neighborhood    #
    #    -INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood    #
    img_color = cv.resize(img_color, dsize=(width,height), interpolation=cv.INTER_CUBIC)
    
    # Return output frame
    return np.int64(img_color)
#=============================================================================#

#=============================================================================#
#      Node statistcs                                                         #
#=============================================================================#
def nodeStats(img, point=None):
    '''
    Function evaluates the summary of the product each colour channel with its
    corresponding transposed colour channel.
    
    If a point is entered, the function performs the same operations, but on
    a point specified area of the input frame.
    
    Parameters
    ----------
    img : int64 multidimensional array
        Input frame.
        
    point : 
        DESCRIPTION.
        
    Returns
    -------
    TYPE : int64
        DESCRIPTION.
    
    '''
    # Treat complete image as single cluster and compute different fields of
    # cluster
    R = np.zeros((3, 3), dtype=np.int64)
    
    # If point present
    if point is not None:
        for i in range(0, 3):
            for k in range(0, 3):
                R[i, k] = np.sum(np.multiply(img[point, i], img[point, k]),
                                 axis=0)
    # If point not present
    else:
        
        for i in range(0,3):
            for k in range(0, 3):
                R[i, k] = np.sum(np.sum(np.multiply(np.matrix(img[:, :, i]),
                                np.matrix(img[:, :, k])), axis=0))
    
    # Return resuting operatins
    return np.int64(R)
#=============================================================================#

#=============================================================================#
#       Image components                                                      #
#=============================================================================#
def img_comps_proc(img, point=None):
    '''
    
    
    Parameters
    ----------
    img : int64 multidimensional array.
        Normalized input Frame.
    point : 
        DESCRIPTION.
    
    Returns
    -------
    img_comps : int64 multidimensional array.
        DESCRIPTION.
    
    '''
    # Frame is RGB
    try:
        height, width, col = img.shape
        
    # Frame is Grayscale
    except ValueError:
        height, width = img.shape
        col = 1
        
    # Check if point not provided
    if point is None:
        img_comps = np.zeros((height * width, col), dtype=np.int64)
        for i in range(col):
            img_comps[:, i] = np.ravel(img[:, :, i])
            
    # If point provided
    else:
        # TODO: fix magic numbers
        img_comps = np.zeros((1, 3), dtype=np.int64)
        
        for i in range(3):
            img_comps[:, i] = np.sum(img[point, i], axis=0)
    
    # Return image componenets
    return np.int64(img_comps)
#=============================================================================#


#=============================================================================#
# Covariance function
#=============================================================================#
def eigenVec(img, R, height, width):
    '''
    Covariance function of ???

    Inputs:       R: ???
                  Npix: ???
                  Comp_vec: ???
                  img_norm:  Normalized frame

    Outputs:      V: Eigen values of ???
    '''
    
    # Number of pixels
    Npix = np.int(height * width)
    
    # Component vector
    Comp_vec = sum_Comp(img)
    
    # What is this?
    R_bar = R - (np.dot(Comp_vec, Comp_vec.T) / Npix)
    
    # Compute eigenvector of 
    _, eigenvector = LA.eigh(R_bar)
    
    # Return eigenvector
    return eigenvector[:, 1]
#=============================================================================#

#=============================================================================#
# Transpose function
#=============================================================================#
def quan_proc(height, width, img_in):
    '''
    Function evaluates the mean (quantization) value of the input frame/node.
    
    Parameters
    ----------
    height : int
        DESCRIPTION.
    width : int
        DESCRIPTION.
    img_in : int64 multidimensional array
        Input Frame/Node.
    
    Returns
    -------
    TYPE : flaot
        Quantization (mean) value of input Frame/Node.
    
    '''
    
    # Number of pixels of the specific node
    Npix = np.int(height * width)
    
    # Component vector
    Comp_vec = sum_Comp(img_in)
    
    # Return quantization (mean) value of input.
    return np.transpose((Comp_vec / Npix))

#=============================================================================#

#=============================================================================#
# Sum Comp_vec Function
#=============================================================================#
def sum_Comp(img):
    '''
    Function returns a 1x3 component vector with the total sum of each
    colour channel seperately.
    
    Parameters
    ----------
    img : int64 multidimensional array
        Input frame.
        
    Returns
    -------
    Comp_vec : int64 1x3 array
        Component vector of input frame's present colour channel(s).
        
    '''
    
    # Assert if framne is BGR
    try:
        height, width, col = img.shape
        
    # Frame is Grayscale
    except ValueError:
        col = 1
    
    # Get input's data type for output consistency
    dtype = img.dtype
    
    # Initialize output variable
    Comp_vec = np.zeros(col, dtype)
    
    # Evaluate sum of each present colour channel
    for i in range(col):
        Comp_vec[i] = np.sum(np.sum(np.matrix(img[:, :, i]), axis=0))
        
    # Return component vector
    return Comp_vec
#=============================================================================#

#=============================================================================#
# Point function
#=============================================================================#
def point_proc(vector, img_comps, Q, point):
    # TODO: Complete docstring, implement control "else" statement (?)
    '''


    Parameters
    ----------
    vector : TYPE
        DESCRIPTION.
    img_comps : TYPE
        DESCRIPTION.
    Q : 
        Quantization (mean value) of node.
    point : string
        Input flag to return corresponding point.

    Returns
    -------
     : TYPE
        DESCRIPTION.

    '''
    # Change var name, what does it do???
    a = np.dot(vector[0],img_comps[:, 0]) + np.dot(vector[1],
              img_comps[:, 1]) + np.dot(vector[2],img_comps[:, 2])

    if point == 'Point1':
        return (np.arange( len(a) )[ a >= np.dot(vector.T, Q.T)])

    elif point == 'Point2':
        return(np.arange( len(a) )[ a < np.dot(vector.T, Q.T)])
    # User entered invalid option
    else:
        print("Invalid option entered.")
        return -1

#=============================================================================#

# =========================================================================== #
# Some eigen function
# =========================================================================== #
def eigen_function(img, img_comp, height, width, point, clst_flg, out_flg):
    # TODO: Introduce comments; change variable names. Fill docstring.
    # Change list to numpy object (e.g. array)
    # WARNING!!! WILL RETURN TUPLE IF NOT USING "np.array" for main var
    '''
    
    '''
    # Number of pixels in Node
    Npix = np.size(point)
    
    # If desired 
    if clst_flg == 0:
        # Component vector of Node
        Comp_vec = img_comps_proc(img_comp, point)
        
        # Node statistics
        R = nodeStats(img_comp, point)
        
        
    # Else if desired *to-be-filled*
    #elif clst_flg == 1:
        # Comment here
        #R = nodeStats(img, None) - nodeStats(img_comp, point)
        
        # Comment here
        #Comp_vec = sum_Comp(img) - img_comps_proc(img_comp,  point)
        
        
    # Difference between covariance & Node statistics
    R_bar = R - ((np.dot(Comp_vec, Comp_vec.T)) / Npix)
    
    # Evaluate eigen values & vector on above difference
    D, V = LA.eigh(R_bar)
    
    # Output of cluster
    if out_flg == 'cluster':
        
        matrix = np.zeros((height,width,3), dtype=np.int64)
        y_s, x_s = np.unravel_index(point,(height,width),'C')
        
        for i in range(0,Npix):
            y = y_s[i]
            x = x_s[i]
            matrix[y,x,0] = img[y,x,0]
            matrix[y,x,1] = img[y,x,1]
            matrix[y,x,2] = img[y,x,2]
            
            
        # Return array of eigenvectors for each node, along with node
        return np.array((np.float64(V[:,1]), np.int64(matrix), np.float64(V[:,0]),
                          np.float64(V[:,2])))
    
    # Output of eigenvalues
    elif  out_flg == 'eig':
        return D
    # User entered invalid flag
    else:
        print("Invalid option entered")
        return -1
# =========================================================================== #

# =========================================================================== #
# Eigen function
# =========================================================================== #
def eig_vals(img, img_comps, height, width, point1, point2, index, si, ret_flg):
    '''

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    img_comps : TYPE
        DESCRIPTION.
    height : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    point : TYPE
        DESCRIPTION.
    ret_flg : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    
    if ret_flg == 'eig_val':
        # Store in position 1
        eig_val[index] = np.abs(eigen_function(img, img_comps,
               height, width, point1, 0,'eig')[1])
        # Store in position 2
        eig_val[si] = np.abs(eigen_function(img, img_comps,
               height, width, point2, 0,'eig')[1])
        # Return resulting array
        return eig_val
    
    elif ret_flg == 'eig_val_l':
        # Store in position 1
        eig_val_l[index] = eigen_function(img, img_comps, height,
                                         width, point1, 0,'eig')[0]
        # Store in position 2
        eig_val_l[si] = eigen_function(img, img_comps, height,
                                    width, point2, 0,'eig')[0]
        # Return resulting array
        return eig_val_l
    
    elif ret_flg == 'eig_v_l':
        # Store in position 1
        eig_v_l[index] = eigen_function(img, img_comps, height,
                                       width, point1, 0,'eig')[0]
        # Store in position 2
        eig_v_l[si] = eigen_function(img, img_comps, height,
                                    width, point2, 0,'eig')[0]
        # Return resulting array
        return eig_v_l
    else:
        print("Invalid option entered.")
        return -1

# =========================================================================== #


def split_nodes(img, height, width, K):
    '''
    
    Parameters
        img - the input image of height and width
        K - number of clusters / nodes
    Returns
        Clusters C with eigen
    '''
    # Node statistics, eigenvector & components
    R = nodeStats(img)
    img_comps = img_comps_proc(img)
    vector = eigenVec(img, R, height, width)
    
    # Node quantization value & components
    Comp_vec = sum_Comp(img)
    Q = quan_proc(height, width, img)
    
    # Pixels of the cluster are divided into two parts based on their closeness -
    # to a plane perpendicular to principal eigen vector and passing through mean.
    # Nodes stored at structure C
    point1 = point_proc(vector, img_comps, Q, point='Point1')
    point2 = point_proc(vector, img_comps, Q, point='Point2')
    
    C = np.array((eigen_function(img, img_comps, height, width, point1, 0,'cluster'),
                  eigen_function(img, img_comps, height, width, point2, 0,'cluster')),
        ndmin=2)
    
    index = 0
    si = 1
    eig_val = eig_vals(img, img_comps, height, width, point1, point2, index, si, 'eig_val')
    eig_val_l = eig_vals(img, img_comps, height, width, point1, point2, index, si, 'eig_val_l')
    eig_v_l = eig_vals(img, img_comps, height, width, point1, point2, index, si, 'eig_v_l')
    
    # The two nodes are further splitted until K nodes.
    for i in range (2,K):
         if (np.max(eig_val) == -1):
             break
         
         # Pick the node having highest Eigen value and split it into two parts.
         index = np.int(eig_val.argmax())
         Matrix = C[index, 1]
         vector = C[index, 0]
         R = nodeStats(Matrix)
         R_comp = img_comps_proc(Matrix)
         Comp_vec = sum_Comp(Matrix)
         Q = quan_proc(height, width, Matrix)
         
         ## Pixels of the cluster are divided into two parts based on their
         ## closeness to a plane perpendicular to principal eigen vector
         ## and passing through mean.
         
         point1 = point_proc(vector, R_comp, Q, 'Point1')
         point2 = point_proc(vector, R_comp, Q, 'Point2')
         
         # Compute various statistical value for the newly formed clusters.
         
         R1 = nodeStats(R_comp, point1)
         M1 = img_comps_proc(R_comp, point1)
         R2 = R - R1
         M2 = Comp_vec - M1
         N1 = np.size(point1)
         N2 = np.size(point2)
         
         if (N1 > 0 and N2 > 0):
             
             # Pad array to store clusters
             C = np.pad(C, ((0, 1), (0, 0)), 'constant')
             
             # Pad array to store eigen values
             
             eig_val = np.pad(eig_val, ((0 , 1)), 'constant')
             
             eig_val_l = np.pad(eig_val_l, ((0, 1)), 'constant')
             
             eig_v_l = np.pad(eig_v_l, ((0, 1)), 'constant')
             
             
             C[index, :] = eigen_function(Matrix, R_comp, height, width, point1, 0,
                                       'cluster')
             
             C[si+1, :] = eigen_function(Matrix, R_comp, height, width, point2, 0,
                                       'cluster')
             
             
             si += 1
             
             eig_val = eig_vals(Matrix, R_comp, height, width, point1,
                                         point2, index, si, 'eig_val')
             
             eig_val_l = eig_vals(Matrix, R_comp, height, width, point1,
                                               point2, index, si, 'eig_val_l')
             
             eig_v_l = eig_vals(Matrix, R_comp, height, width, point1,
                                          point2, index, si, 'eig_v_l')
         else:
             eig_val[index] = - 1
             i -= 1
             
    return (C, eig_val, eig_val_l, eig_v_l)

#=============================================================================#



#=============================================================================#
#    4 CONDITIONS TO CHOSE THE CORRECT NODE FUNCTION                          #
#=============================================================================#
def correct_node(eig_left_sum, eig_right_sum, eig_val_l,
                 eig_r, eig_mid, eig_l, eig_v_l, eig_m):
    # TODO: Introduce appropraite comments; change variable names
    # Preallocate ind variable (?). Change function name?
    '''
    Function that detects the appropriate index of the desired node (?)
    
    Inputs:                 eig_left_sum_idx:
        
                            eig_right_sum_idx:
                                
                            eig_val_l_idx:
                                
    Outputs:                ind:
    '''
    
    # #=============================================================================#
    # #       CHOOSE THE CORRECT NODE                                               #
    # #=============================================================================#
     # The first node is not taken into consideration to avoid wrong decision of
     # the important node, for that reason i give these values 0, 0 and 3
     # (i set a neutral value just to preserve the position in order to be the
     # list as long as the number of nodes)
    for i in range(0, K):
         if i < 1: # just to avoid the first node:
             eig_mid[i] = 0  # set neutral value for first node
             eig_left_sum[i] = 0 # set neutral for 1st node
             eig_right_sum[i] = 3 # set neutral for 1st node
             eig_m[i] = C[i, 0][1] # save value for 1st node
             eig_l[i] = np.sum(C[i, 2]) # save value for 1st node
             eig_r[i] = np.sum(C[i, 3]) # save value for 1st node
             
         else:  # continue from second node:
             eig_mid[i] = np.abs(C[i, 0][1])
             eig_m[i] = C[i, 0][1] # here no abs needed
             eig_left_sum[i] = np.sum(C[i, 2])
             eig_l[i] = np.sum(C[i, 2])
             eig_right_sum[i] = np.abs(np.sum(C[i, 3]))
             eig_r[i] = np.sum(C[i, 3]) # here no abs needed
             
             
     # same for the statistic eig_val_l, i avoid the first two nodes in order -
     # to increase the possibility of success:
    eig_val_l[0] = 0    # give neutral value for 1st node.
    eig_val_l[1] = 0    # give neutral value for 2nd node.
    
    
    # #=============================================================================#
    # #           CONDITION LOOP IN CASE THE NODE IS TOTALLY BLACK                  #
    # #=============================================================================#
     # When the saved node is black in these 3 statistics the saved values are 1
     # for that reason i have to make it 0 to not confuse my indexes and pickup
     # one of the black nodes:
    for i in range(0, K):
         if eig_mid[i] == 1:
             eig_mid[i] = 0
         if eig_left_sum[i] == 1:
             eig_left_sum[i] = 0
         if eig_right_sum[i] == 1:
             eig_right_sum[i] = 0
             
    # find the node with MAX mid Eigen Value of the Vector
    eig_mid_index = np.int(eig_mid.argmax())
    
    # find the node with MAX sumary of left Eigen Vector
    eig_left_sum_index = np.int(eig_left_sum.argmax())
    
    # find the node with MIN  sumary of right Eigen Vector
    eig_right_sum_index = np.int(eig_right_sum.argmin())
    
    # find the node with MIN left Eigen Value
    eig_val_l_index = np.int(eig_val_l.argmin())
    
    # Indexes created for use in 4th condition.....#
    eig_r_in = np.int(eig_r.argmin())
    eig_l_in = np.int(eig_l.argmin())
    eig_v_l_in = np.int(eig_v_l.argmax())
    eig_m_in = np.int(eig_m.argmax())
    
    # Comment
    if eig_left_sum_index == eig_right_sum_index == eig_val_l_index: # 1st 37%
        ind = eig_mid_index
        
    # Comment
    elif eig_left_sum_index == eig_mid_index: #======================= 2nd 46%
        ind = eig_mid_index
        
    # Comment
    elif eig_left_sum_index == eig_val_l_index: #===================== 3rd 4%
        ind = eig_val_l_index
        
    # Comment
    elif eig_left_sum_index == eig_right_sum_index: #================= 4th 9%
        # Comment
        if (eig_r_in == eig_l_in == eig_v_l_in) & (eig_m_in == eig_mid_index):
            ind = eig_r_in
            
        # Comment
        else:
            ind = eig_mid_index
    else:
        ind = eig_left_sum_index #====================================  2%
        
    # Return node
    return ind
#=============================================================================#



# Example implementation...
#=============================================================================#
if __name__ == '__main__':
#=============================================================================#
    img_path = get_path('*.jpg')
    img = cv.imread(img_path)
    height = 1080
    width = 1920
    img = frame_proc(img, height, width)

    #=============================================================================#
    #           CUSTOM:        250x440 -   1.6 sec per image                      #
    #           CUSTOM:        400x700 -     4 sec per image                      #
    #           CUSTOM:       800x1400 -    18 sec per image                      #
    #        REALSENSE:      1080x1920 -    32 sec per image                      #
    #           IPHONE:      3024x4032 -  3:16 min per image                      #
    #=============================================================================#

    K = 6

    C = np.zeros(shape=(K,4))
    eig_val = np.zeros((K))
    eig_val_l = np.zeros((K))
    eig_v_l = np.zeros((K))
    eig_mid = np.zeros((K))
    eig_left_sum = np.zeros((K))
    eig_right_sum = np.zeros((K))
    eig_m = np.zeros((K))
    eig_l = np.zeros((K))
    eig_r = np.zeros((K))

    # Processing
    C, eig_val, eig_val_l, eig_v_l = split_nodes(img, height, width, K)
    ind = correct_node(eig_left_sum, eig_right_sum,
                       eig_val_l, eig_r, eig_mid,
                       eig_l, eig_v_l, eig_m)
    node = C[ind][1]

    # Visualisation
    for j in range(0,K,1):
        temp = C[j][1]
        plt.figure()
        plt.imshow(temp)
        plt.show()
        print("\n [ Node {0} ]".format(str(j+1)))

    plt.imshow(img)
    plt.show()
    print("Original Image")
    plt.imshow(node)
    plt.show()
    print("Best Node Image")

###
