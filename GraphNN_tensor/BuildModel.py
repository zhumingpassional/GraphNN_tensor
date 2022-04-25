#         Build train Model
#           Xiaochen  Han
#            Apr 26 2019
#    guillermo_han97@sjtu.edu.cn
#

import tensorflow as tf
import numpy as np
import LoadData as LD
import DefineParam as DP
# import tensorflow_probability as tfp
# from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.python.framework import ops

# Build Model
def build_model(phi, n, restore=False):
    # tf.reset_default_graph()
    ops.reset_default_graph()

    # Get param
    pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, missingNum, trainFile, testFile, maskFile, saveFile, modelDir = DP.get_param(n)
    
    # pre-process
    Xinput, Xoutput, Phi, PhiC, phiInd, Yinput = LD.pre_calculate(phi, n)

    # build model
    prediction = build_ConvGTN(Xinput, Phi, PhiC, Yinput, nPhase, nFrame)

    # loss function
    costMean = compute_cost(prediction, Xoutput, phiInd, nPhase)
    costAll = costMean
    optmAll = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(costAll)

    # set up
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    sess = tf.Session(config=config)

    if restore is False:                                       # training
        sess.run(init)
        return sess, saver, Xinput, Xoutput, costAll, optmAll, Yinput, prediction, phiInd
    else:                                                      # reconstruction
        saver.restore(sess, '%s/%d.cpkt' % (modelDir, ncpkt))
        return sess, saver, Xinput, Xoutput, Yinput, prediction, phiInd


# Add weight
def get_filter(wShape, nOrder):
    weight = tf.get_variable(shape=wShape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='weight_%d' % (nOrder))
    return weight

# build one phase
def build_one_phase_old(layerxn, Phi, PhiC, Yinput, nFrame):
    # params
    softThr = tf.Variable(0.1, dtype=tf.float32)
    convSize1 = 64
    convSize2 = 64
    convSize3 = 64
    filterSize1 = 3
    filterSize2 = 3
    filterSize3 = 3 

    # get rn from xn-1
    rn = tf.add(Yinput, tf.multiply(layerxn[-1], PhiC))

    # FC(rn)
    rn = tf.layers.dense(rn, 100)

    # F(FC(rn))
    weight0 = get_filter([filterSize1, filterSize1, nFrame, convSize1], 0)
    weight1 = get_filter([filterSize2, filterSize2, convSize1, convSize2], 1)
    weight2 = get_filter([filterSize3, filterSize3, convSize2, convSize3], 2)
    Frn = tf.nn.conv2d(rn, weight0, strides=[1, 1, 1, 1], padding='SAME')
    Frn = tf.nn.conv2d(Frn, weight1, strides=[1, 1, 1, 1], padding='SAME')
    Frn = tf.nn.relu(Frn)
    Frn = tf.nn.conv2d(Frn, weight2, strides=[1, 1, 1, 1], padding='SAME')

    # soft(F(FC(rn)))
    softFrn = tf.multiply(tf.sign(Frn), tf.nn.relu(tf.subtract(tf.abs(Frn), softThr)))

    # zn = ~F(soft(F(FC(rn)), softThr))
    weight3 = get_filter([filterSize3, filterSize3, convSize3, convSize2], 3)
    weight4 = get_filter([filterSize2, filterSize2, convSize2, convSize1], 4)
    weight5 = get_filter([filterSize1, filterSize1, convSize1, nFrame], 5)
    zn = tf.nn.conv2d(softFrn, weight3, strides=[1, 1, 1, 1], padding='SAME')
    zn = tf.nn.relu(zn)
    zn = tf.nn.conv2d(zn, weight4, strides=[1, 1, 1, 1], padding='SAME')
    zn = tf.nn.conv2d(zn, weight5, strides=[1, 1, 1, 1], padding='SAME')

    # xn = rn + ~F(soft(F(FC(rn)), softThr))
    xn = tf.add(rn, zn)

    return xn

def build_one_phase(layerxn, Phi, PhiC, Yinput, nFrame):
    # params
    softThr = tf.Variable(0.1, dtype=tf.float32)
    convSize1 = 100
    convSize2 = 100
    filterSize1 = 3
    filterSize2 = 3

    # get rn from xn-1
    rn = tf.add(Yinput, tf.multiply(layerxn[-1], PhiC))

    # F(FC(rn))
    rn = tf.layers.dense(rn, 100)
    weight0 = get_filter([filterSize1, filterSize1, nFrame, convSize1], 0)
    weight1 = get_filter([filterSize2, filterSize2, convSize1, convSize2], 1)
    Frn = tf.nn.conv2d(rn, weight0, strides=[1, 1, 1, 1], padding='SAME')
    Frn = tf.nn.relu(Frn)
    Frn = tf.nn.conv2d(Frn, weight1, strides=[1, 1, 1, 1], padding='SAME')

    # soft(F(FC(rn)))
    softFrn = tf.multiply(tf.sign(Frn), tf.nn.relu(tf.subtract(tf.abs(Frn), softThr)))

    # zn = ~F(soft(F(FC(rn)), softThr))
    weight2 = get_filter([filterSize2, filterSize2, convSize2, convSize1], 2)
    weight3 = get_filter([filterSize1, filterSize1, convSize1, nFrame], 3)
    zn = tf.nn.conv2d(softFrn, weight2, strides=[1, 1, 1, 1], padding='SAME')
    zn = tf.nn.relu(zn)
    zn = tf.nn.conv2d(zn, weight3, strides=[1, 1, 1, 1], padding='SAME')
    zn = tf.layers.dense(zn, 100)

    # xn = rn + ~F(soft(F(FC(rn)), softThr))
    xn = tf.add(rn, zn)

    return xn


# compute fista once (one epoch)
def build_ConvGTN(Xinput, Phi, PhiC, Yinput, nPhase, nFrame):
    layerxn = []
    layerxn.append(Xinput)                            # x0 = x0
    for i in range(nPhase):
        with tf.variable_scope('conv_%d' % (i), reuse=False):
            xn = build_one_phase(layerxn, Phi, PhiC, Yinput, nFrame)
            layerxn.append(xn)
    return layerxn


# compute loss function
def compute_cost(prediction, Xoutput, phiInd, nPhase):
    costMean = 0
    #for j in range(nPhase):
    for i in range(len(phiInd)):
        costMean += tf.reduce_mean(tf.square(prediction[-1][:, :, :, phiInd[i]] - Xoutput[:, :, :, phiInd[i]]))
    return costMean

