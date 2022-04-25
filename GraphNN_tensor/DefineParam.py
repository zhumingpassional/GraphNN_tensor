#          Define all param
#           Xiaochen  Han
#            May 21 2019
#    guillermo_han97@sjtu.edu.cn
#

mNum = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50] 

# define all param
def get_param(n):    
    pixel = 24
    batchSize = 32
    nPhase = 10
    nTrainData = 900
    trainScale = 1              # scale of training part and validating part
    learningRate = 0.0001
    nEpoch = 500
    nFrame = 100
    ncpkt = nEpoch

    missingNum = mNum[n]

    trainFile = './trainData/train1.mat'
    testFile = './testData/test1.mat'
    maskFile = './maskData/mask_%d.mat' % missingNum
    saveFile = './RecGraph/result_%d' % missingNum
    modelDir = './Model/missingNum_%d' % missingNum
    
    #trainFile = './trainData/train_20.mat'
    #maskFile = './maskData/mask.mat'
    #modelDir = './Model50'
    #saveFile = 'result'

    return pixel, batchSize, nPhase, nTrainData, trainScale, learningRate, nEpoch, nFrame, ncpkt, missingNum, trainFile, testFile, maskFile, saveFile, modelDir



