#          Load train data
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import tensorflow as tf
import scipy.io as sio
import numpy as np
import DefineParam as DP
import h5py


# Load training data
def load_train_data(n, mat73=False):
    # Get param
    pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, missingNum, trainFile, testFile, maskFile, saveFile, modelDir = DP.get_param(n)
    
    if mat73 == True:                                                # if .mat file is too big, use h5py to load
        trainData = h5py.File(trainFile)
        trainLabel = np.transpose(trainData['trainNodes'], [3, 2, 1, 0])
    else:
        trainData = sio.loadmat(trainFile)
        trainLabel = trainData['trainNodes']                             # labels

    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']                                            # mask

    del trainData, maskData
    return trainLabel, phi


# Load testing data
def load_test_data(n, mat73=False):
    # Get param
    pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, missingNum, trainFile, testFile, maskFile, saveFile, modelDir = DP.get_param(n)

    print("mask file: %s" % maskFile)
    
    if mat73 == True:
        testData = h5py.File(testFile)
        testLabel = np.transpose(testData['testNodes'], [3, 2, 1, 0])
    else:
        testData = sio.loadmat(testFile)
        testLabel = testData['testNodes']
    
    maskData = sio.loadmat(maskFile)
    phi = maskData['phi']

    del testData, maskData
    return testLabel, phi


# Compute essential variables
def pre_calculate(phi, n):
    # Get param
    pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, missingNum, trainFile, testFile, maskFile, saveFile, modelDir = DP.get_param(n)

    Xinput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])      # X0
    Xoutput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])     # labels
    Yinput = tf.placeholder(tf.float32, [None, pixel, pixel, nFrame])      # measurement
    Phi = tf.constant(phi)
    PhiC = tf.constant(1.0 - phi, dtype=tf.float32)
    phiInd = []
    for i in range(nFrame):
        if phi[1,1,i] == 0:
            phiInd.append(i)
    print(phiInd)

    return Xinput, Xoutput, Phi, PhiC, phiInd, Yinput













