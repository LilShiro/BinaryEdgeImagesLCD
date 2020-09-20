import numpy as np
import torch
import torchvision
import cv2
import torch.nn as nn
import random
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve
from scipy.spatial.distance import euclidean

# Pytorch transformation function class to transform grayscale image to binary edge images
# Arguments: output_size - A tuple containing dimensions to transform initial image.
#            threshold - Real Value representing threshold gradient values.
# Output: binary edge images of PIL images transformation is applied to
class getBinEdge(object):
    def __init__(self, output_size, threshold):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.threshold = threshold

    def __call__(self, sample):
        # resizing initial image
        expand = cv2.resize(np.array(sample), self.output_size)
        expand = np.expand_dims(expand, 0)
        expand = np.expand_dims(expand, 0)
        tensor = torch.tensor(expand, dtype=torch.long)

        # defining sobel arrays
        sobel_x_rev = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        sobel_y_rev = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
        sobel_x_rev = np.expand_dims(sobel_x_rev, 0)
        sobel_x_rev = np.expand_dims(sobel_x_rev, 0)
        sobel_y_rev = np.expand_dims(sobel_y_rev, 0)
        sobel_y_rev = np.expand_dims(sobel_y_rev, 0)

        # obtain image gradient with sobel filters
        conv_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        conv_x.weight = nn.Parameter(torch.tensor(sobel_x_rev, dtype=torch.long), requires_grad=False)
        image_grad_x = conv_x(tensor)[0][0]
        conv_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        conv_y.weight = nn.Parameter(torch.tensor(sobel_y_rev, dtype=torch.long), requires_grad=False)
        image_grad_y = conv_y(tensor)[0][0]
        image_grad = np.sqrt((np.square(image_grad_x) + np.square(image_grad_y)))

        # thresholding
        image_grad = np.where(image_grad > self.threshold, 1, 0)
        return image_grad

# Function to apply Random Homographic Deformation on images
# Arguments: Images - Images to be randomly deformed
# Returns: Randomly deformed images
def addNoise(images):
    full = []
    for i in images:
        # random coin flip to determine if we apply transform
        if bool(random.getrandbits(1)):
            full.append(i.type(torch.double))
        else:
            i_c = np.array(i[0]) * 255
            random_ints = np.random.randint(64, size=8)
            # Pick random points within 4 corners of images
            topleft = random_ints[:2]
            topright = np.array([random_ints[2] + 191, random_ints[3]])
            bottomleft = np.array([random_ints[4], random_ints[5] + 191])
            bottomright = np.array([random_ints[6] + 191, random_ints[7] + 191])

            # apply 2D homography transformation
            destination = np.vstack((topleft, topright, bottomright, bottomleft))
            source = np.array([[0, 0], [255, 0], [255, 255], [0, 255]])
            h = cv2.getPerspectiveTransform(np.float32(destination), np.float32(source))
            final = cv2.warpPerspective(np.float32(i_c), h, (256, 256))

            # convert back to format for neural network processing
            final = torch.tensor(final, dtype=torch.double).view(1, 256, 256)
            full.append(final / 255)
    return torch.cat(full).view(-1, 1, 256, 256)

# Function to compute similarity matrix
# Arguments: full_latent - matrix of all compiled latent vectors stacked by rows
# Return: Similarity Matrix (Euclidean Distance)
def calc_sim(full_latent):
    rowNo = full_latent.shape[0]
    sim_matrix_MSE = np.zeros((full_latent.shape[0], full_latent.shape[0]))
    for row1 in range(rowNo):
        for row2 in range(row1 + 1):
            sim_matrix_MSE[row1][row2] = euclidean(full_latent[row1], full_latent[row2])
            sim_matrix_MSE[row2][row1] = sim_matrix_MSE[row1][row2]
    sim_matrix_MSE = 10 - np.log(sim_matrix_MSE + 1)
    sim_matrix_MSE -= (np.min(sim_matrix_MSE)) - 1e-9
    sim_matrix_MSE /= np.max(sim_matrix_MSE)
    return sim_matrix_MSE


# Function to do rank reduction
# Arguments: sim_matrix - Matrix to apply rank reduction on
#            reductionvalue - degree of rank reduction
#            img_apart - minimum gap between loop closure frames
# Return: Rank Reduced Matrix
def rankreduction(sim_matrix, reductionvalue, img_apart):
    if reductionvalue == 0:
        # return same matrix
        rankreduced = sim_matrix.copy()
    else:
        eigenvalues, eigenvectors = eigh(sim_matrix)
        rankreduced = sim_matrix.copy()
        # reduce based on reduction value specified
        for i in range(1, reductionvalue + 1):
            rankreduced -= eigenvalues[-i] * (np.expand_dims(eigenvectors[:, -i], axis=1).dot(np.expand_dims(eigenvectors[:, -i], axis=0)))
    # normalise values based on gap specified
    rankreduced -= np.min(np.tril(rankreduced, -img_apart))
    rankreduced /= np.max(np.tril(rankreduced, -img_apart))
    rankreduced = np.where(rankreduced > 1, 1, rankreduced)
    rankreduced = np.where(rankreduced < 0, 0, rankreduced)
    return rankreduced

# Function to do smith-water in one direction (moving down and towards the right)
# Arguments: rankreduced_smith - Matrix to apply smith_waterman on
#            delta - penalty term for many-to-one matching
#            min_seq_sim - minimum sequence similarity required
# Return:    smithsonian - one way smith matrix
#            min_tracker_seq - sequence of minimum similarity sequence that passes threshold
#            sw_matrix_min - cumulative similarity score for minimum similarity sequence that passes threshold
#            max_tracker_seq - sequence of maximum similarity sequence that passes threshold
#            sw_matrix_max - cumulative similarity score for maximum similarity sequence that passes threshold
def smith_water(rankreduced_smith, delta, min_seq_sim):
    # tabulate cumulative matrix
    tracker = np.empty(rankreduced_smith.shape, dtype=object)
    sw_matrix = np.zeros(rankreduced_smith.shape)
    for i in range(rankreduced_smith.shape[0]):
        for j in range(rankreduced_smith.shape[0]):
            if i == 0:
                maxindex = 2
                maxval = sw_matrix[i][j - 1]
            elif j == 0:
                maxindex = 1
                maxval = sw_matrix[i - 1][j]
            else:
                maxindex = np.argmax(np.array([sw_matrix[i - 1][j - 1], sw_matrix[i - 1][j] - delta, sw_matrix[i][j - 1] - delta]))
                maxval = np.amax(np.array([sw_matrix[i - 1][j - 1], sw_matrix[i - 1][j] - delta, sw_matrix[i][j - 1] - delta]))

            if maxindex == 2:
                sw_matrix[i][j] = sw_matrix[i][j - 1] + rankreduced_smith[i][j] - delta
                tracker[i][j] = (i, j - 1)
            elif maxindex == 1:
                sw_matrix[i][j] = sw_matrix[i - 1][j] + rankreduced_smith[i][j] - delta
                tracker[i][j] = (i - 1, j)
            else:
                sw_matrix[i][j] = sw_matrix[i - 1][j - 1] + rankreduced_smith[i][j]
                tracker[i][j] = (i - 1, j - 1)
            if maxval <= 0:
                tracker[i][j] = None
                sw_matrix[i][j] = rankreduced_smith[i][j]
            if rankreduced_smith[i][j] < 0:
                tracker[i][j] = None
                sw_matrix[i][j] = 0

    # obtain all sequences that pass the cumulative similarity score
    smithsonian = np.zeros(rankreduced_smith.shape)
    nodes = np.argwhere(sw_matrix > min_seq_sim)
    sw_matrix_max = 0
    sw_matrix_min = float('inf')
    if nodes.shape == (0, 2):
        return smithsonian, [], sw_matrix_min, [], sw_matrix_max
    for node in nodes:
        smithtrack = node
        if sw_matrix[node[0], node[1]] > sw_matrix_max:
            sw_matrix_max = sw_matrix[node[0], node[1]]
            max_tracker = node
        if sw_matrix[node[0], node[1]] < sw_matrix_min:
            sw_matrix_min = sw_matrix[node[0], node[1]]
            min_tracker = node
        while np.all(smithtrack) != None and smithsonian[smithtrack[0], smithtrack[1]] == 0:
            smithsonian[smithtrack[0], smithtrack[1]] = 1
            smithtrack = tracker[smithtrack[0], smithtrack[1]]

    # track maximum and minimum sequence that passes sequence threshold
    max_tracker_seq = []
    smithmaxtrack = tuple(max_tracker)
    while np.all(smithmaxtrack) != None:
        max_tracker_seq.append(smithmaxtrack)
        smithmaxtrack = tracker[smithmaxtrack[0], smithmaxtrack[1]]
    min_tracker_seq = []
    smithmintrack = tuple(min_tracker)
    while np.all(smithmintrack) != None:
        min_tracker_seq.append(smithmintrack)
        smithmintrack = tracker[smithmintrack[0], smithmintrack[1]]

    return smithsonian, min_tracker_seq, sw_matrix_min, max_tracker_seq, sw_matrix_max

# Full Smith-Waterman Algorithm (in both directions)
# Arguments: rankreduced - matrix to apply smith-waterman on
#            delta - penalty term for many-to-one matching
#            min_seq_sim - minimum sequence similarity required
#            min_sim - minimum similarity threshold for individual image pairs
#            imgapart - minimum gap between loop closure frames
# Return:    full_smithsonian - two way final smith matrix
#            min_tracker_full - sequence of overall minimum similarity sequence that passes threshold
#            max_tracker_full - sequence of overall maximum similarity sequence that passes threshold
def full_smith_water(rankreduced, delta, min_seq_sim, min_sim, imgapart):
    # flip matrix and do smith waterman in each direction
    rankreduced_smith = np.where(np.tril(rankreduced, -imgapart) > min_sim, rankreduced, -2)
    smithsonian, min_tracker_seq, sw_matrix_min, max_tracker_seq, sw_matrix_max = smith_water(rankreduced_smith, delta, min_seq_sim)
    rankreduced_smith_flip = np.flipud(np.where(np.tril(rankreduced, -imgapart) > min_sim, rankreduced, -2))
    smithsonian_flip, min_tracker_seq_flip, sw_matrix_min_flip, max_tracker_seq_flip, sw_matrix_max_flip = smith_water(rankreduced_smith_flip, delta, min_seq_sim)

    # return no loop closure if none found based on conditions specified
    if len(max_tracker_seq_flip) == 0 and len(min_tracker_seq_flip) == 0 and len(max_tracker_seq) == 0 and len(min_tracker_seq) == 0:
        print("No Loop Closure Found")
        return np.zeros(rankreduced.shape), [], []

    # tabulate overall max and min sequence
    if sw_matrix_max > sw_matrix_max_flip:
        max_tracker_full = np.array(max_tracker_seq)
    else:
        max_tracker_seq_flip = np.array(max_tracker_seq_flip)
        max_tracker_seq_flip[:, 0] = np.absolute(max_tracker_seq_flip[:, 0] - rankreduced.shape[0]) - 1
        max_tracker_full = max_tracker_seq_flip

    if sw_matrix_min > sw_matrix_min_flip:
        min_tracker_seq_flip = np.array(min_tracker_seq_flip)
        min_tracker_seq_flip[:, 0] = np.absolute(min_tracker_seq_flip[:, 0] - rankreduced.shape[0]) - 1
        min_tracker_full = np.array(min_tracker_seq_flip)
    else:
        min_tracker_full = min_tracker_seq

    # tabulate compiled final smith waterman matrix
    full_smithsonian = np.flipud(smithsonian_flip) + smithsonian
    full_smithsonian = np.where(full_smithsonian == 2, 1, full_smithsonian)
    return full_smithsonian, min_tracker_full, max_tracker_full

# Function to print matrices
# Arguments: matrix_list - list of prediction matrix to plot PRC curves for
#            matrix_names - list of names for matrices
#            figsize - size of figure to print
# Prints: List of matrices
def printmat(matrix_list, matrix_names, figsize):
    matimage = plt.figure(figsize=figsize)
    plt.tight_layout()
    number = len(matrix_list)
    i = 1
    for matrix in matrix_list:
        matrixsub = matimage.add_subplot(1, number, i)
        matrixsub.set_title(matrix_names[i - 1])
        plt.imshow(matrix.astype(float))
        i += 1

# Function print Precision-Recall Curve
# Arguments: matrix_list - list of prediction matrix to plot PRC curves for
#            full_label - list of labels for images being evaluated on
#            datatype - dataset to perform evaluation on
#            names - list containing name of each matrix within matrix_list
#            img_apart - minimum number of frames between loop closure
#            figsize - size of figure to print
#            x_lim - limit of recall axis
#            smithindex - index within matrix list for smith matrix
#            type - if final smith matrix is being plotted (i.e. truncated curve). This should be the second matrix within matrix_list
# Prints: Precision Recall Curves for various matrix
def PRC(matrix_list, full_label, datatype, names, img_apart, figsize, x_lim, smithindex = 2, type='smith'):
    if datatype == 'newcollege':
        truthlabels = np.genfromtxt('NewCollegeTextFormat.txt', delimiter=',', dtype=int)
    elif datatype == 'citycenter':
        truthlabels = np.genfromtxt('CityCentreTextFormat.txt', delimiter=',', dtype=int)

    prc = plt.figure(figsize=figsize)
    plt.tight_layout()
    j = 0
    # tabulate truth and probability vector
    for rankreduced in matrix_list:
        vectorsize = int((len(full_label) ** 2 - len(full_label)) / 2)
        truthvector = np.zeros(vectorsize)
        probvector = np.zeros(vectorsize)
        i = 0
        for row1 in range(img_apart, len(full_label)):
            for row2 in range(row1 - img_apart):
                truthvector[i] = truthlabels[full_label[row1]][full_label[row2]]
                probvector[i] = rankreduced[row1][row2]
                i += 1

        # plot PRC curve
        precision, recall, threshold = precision_recall_curve(truthvector, probvector)
        max_recall = recall[np.min(np.argwhere(precision == 1))]
        auc_score = round(auc(recall, precision), 3)
        if j == smithindex and type == 'smith':
            plt.plot(recall[1:], precision[1:], label = names[j] + '- Recall at Full Precision: ' + str(max_recall))
        else:
            plt.plot(recall, precision, label = names[j] +'- AUC Score: ' + str(auc_score))
        j += 1
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, x_lim)
    plt.ylim(0, 1.01)
    plt.xticks(np.arange(0, x_lim, step=0.1))
    plt.yticks(np.arange(0, 1.01, step=0.1))
    plt.legend()
    plt.grid()
    plt.title('Precision-Recall Curve')

# Function to plot movement
# Arguments: matrix - Smith-Waterman Matrix
#            datatype - dataset to perform evaluation on
#            full_label - list of labels for images being evaluated on
#            figsize - size of figure to print
# Prints: Predicted Loop Closures vs Actual Loop Closures
def plotmovement(matrix, datatype, full_label, figsize):
    movement = plt.figure(figsize=figsize)
    plt.tight_layout()
    predicted = movement.add_subplot(2, 1, 1)
    predicted.set_title("Predicted Loop Closures")
    if datatype == 'newcollege':
        route = np.genfromtxt("new_college_GPS.txt", dtype=float)
        plt.plot(route[:, 1], route[:, 2])
        lcd = np.argwhere(matrix == 1)
        for point in range(lcd.shape[0]):
            if full_label[lcd[point][0]] < 1984 and full_label[lcd[point][1]] < 1984:
                plt.plot([route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 1]], 1]], [route[full_label[lcd[point, 0]], 2], route[full_label[lcd[point, 1]], 2]],
                         '--g')
                plt.plot(route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 0]], 2], '.r')
                plt.plot(route[full_label[lcd[point, 1]], 1], route[full_label[lcd[point, 1]], 2], '.r')
        truth = movement.add_subplot(2, 1, 2)
        truth.set_title("Truth Label Loop Closures")
        plt.plot(route[:, 1], route[:, 2])
        truth = np.genfromtxt('NewCollegeTextFormat.txt', delimiter=',', dtype=int)[full_label[0]::(full_label[1] - full_label[0]),full_label[0]::(full_label[1] - full_label[0])]
        lcd = np.argwhere(truth == 1)
        for point in range(lcd.shape[0]):
            if full_label[lcd[point][0]] < 1984 and full_label[lcd[point][1]] < 1984:
                plt.plot([route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 1]], 1]], [route[full_label[lcd[point, 0]], 2], route[full_label[lcd[point, 1]], 2]],
                         '--g')
                plt.plot(route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 0]], 2], '.r')
                plt.plot(route[full_label[lcd[point, 1]], 1], route[full_label[lcd[point, 1]], 2], '.r')
    else:
        route = np.genfromtxt("city_center_GPS.txt", dtype=float)
        plt.plot(route[:, 1], route[:, 2])
        lcd = np.argwhere(matrix == 1)
        for point in range(lcd.shape[0]):
            plt.plot([route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 1]], 1]], [route[full_label[lcd[point, 0]], 2], route[full_label[lcd[point, 1]], 2]],
                         '--g')
            plt.plot(route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 0]], 2], '.r')
            plt.plot(route[full_label[lcd[point, 1]], 1], route[full_label[lcd[point, 1]], 2], '.r')
        truth = movement.add_subplot(2, 1, 2)
        truth.set_title("Truth Label Loop Closures")
        plt.plot(route[:, 1], route[:, 2])
        truth = np.genfromtxt('CityCentreTextFormat.txt', delimiter=',', dtype=int)[full_label[0]::(full_label[1] - full_label[0]),full_label[0]::(full_label[1] - full_label[0])]
        lcd = np.argwhere(truth == 1)
        for point in range(lcd.shape[0]):
            plt.plot([route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 1]], 1]], [route[full_label[lcd[point, 0]], 2], route[full_label[lcd[point, 1]], 2]],
                         '--g')
            plt.plot(route[full_label[lcd[point, 0]], 1], route[full_label[lcd[point, 0]], 2], '.r')
            plt.plot(route[full_label[lcd[point, 1]], 1], route[full_label[lcd[point, 1]], 2], '.r')


# Function to plot highest probability image pairs
# Arguments: matrix - prediction matrix to find highest probability pairs
#            dataset - data set to print images from
#            img_apart - minimum number of frames between loop closure
#            datatype - type of data we are working with
#            number - number of top pairs to print
#            title - title of graph
#            figsize - size of figure to print
# Prints: Pairs with highest probability with edge maps
def plothighestpairs(matrix, dataset, datatype, img_apart, number, title, figsize):
    bchange_index1 = np.argsort(np.tril(matrix, -img_apart).flatten())[-number:]
    bchange_image1 = []
    bchange_image1_edge = []
    for index in bchange_index1:
        if datatype != 'scamp':
            bchange_image1.append(dataset.getorig(index // matrix.shape[0]))
            bchange_image1.append(dataset.getorig(index % matrix.shape[0]))
        bchange_image1_edge.append(dataset.__getitem__(index // matrix.shape[0])[0])
        bchange_image1_edge.append(dataset.__getitem__(index % matrix.shape[0])[0])
    if datatype != 'scamp':
        bchange_image1 = torch.stack(bchange_image1, axis=0)
        bchange_image1 = torchvision.utils.make_grid(bchange_image1, 2, 100).numpy().astype(float)
    bchange_image1_edge = torch.stack(bchange_image1_edge, axis=0)
    bchange_image1_edge = torchvision.utils.make_grid(bchange_image1_edge, 2, 100).numpy().astype(float)
    highestpairs = plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.suptitle(title)
    if datatype != 'scamp':
        RGB = highestpairs.add_subplot(1, 2, 1)
        plt.imshow(np.transpose(bchange_image1, (1, 2, 0)))
        edge = highestpairs.add_subplot(1, 2, 2)
        plt.imshow(np.transpose(bchange_image1_edge, (1, 2, 0)))
    else:
        edge = highestpairs.add_subplot(1, 1, 1)
        plt.imshow(np.transpose(bchange_image1_edge, (1, 2, 0)))
# Function to plot maximum and minimum image sequences
# Arguments: sequence - image pair sequence to print
#            dataset - data set to print from
#            datatype - type of data we are working with
#            interval - interval between pairs to print
#            figsize - size of images
#            title - title of figure
# Prints: Pairs of images within sequence along with matrix
def plot_seq(sequence, dataset, datatype, interval, figsize, title):
    img_seq = []
    matrix_ident = np.zeros((len(dataset), len(dataset)))
    i = 0
    for pair in sequence:
        matrix_ident[pair[0], pair[1]] = 1
        if i % interval == 0:
            if datatype == 'scamp':
                img_seq.append(dataset.__getitem__(pair[0])[0])
                img_seq.append(dataset.__getitem__(pair[1])[0])
            else:
                img_seq.append(dataset.getorig(pair[0]))
                img_seq.append(dataset.getorig(pair[1]))
        i += 1
    img_seq = torch.stack(img_seq, axis=0)
    img_seq = torchvision.utils.make_grid(img_seq, 2, 100).numpy().astype(float)
    sampleimgseq = plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.suptitle(title)
    imgseq = sampleimgseq.add_subplot(1, 2, 1)
    plt.imshow(np.transpose(img_seq, (1, 2, 0)))
    matrix = sampleimgseq.add_subplot(1, 2, 2)
    plt.imshow(matrix_ident.astype(float))