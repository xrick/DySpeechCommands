
# coding: utf-8

#import librosa
#import matplotlib
import numpy as np
#import matplotlib.pyplot as plt
import os
import audioUtils
import SpeechModels


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#Convert WAV to Numpy Matrix
def convertWAV2Numpy():
    print('Converting test set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/test/')
    print('Converting training set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/train/')
    return
# prepare the labels for training
basePath = "sd_GSCmdV2"
DysTrainedModelPath = "TrainedModels"

def preparegooglespeechcmd():
    
    GSCmdV2Categs = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 'yes' : 2, 
                         'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11,
                         'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 'six' : 18, 
                         'seven' : 19,  'eight' : 20, 'nine' : 1, 'backward':21, 'bed':22, 'bird':23, 'cat':24, 'dog':25,
                         'follow':26, 'forward':27, 'happy':28, 'house':29, 'learn':30, 'marvin':31, 'sheila':32, 'tree':33,
                         'visual':34, 'wow':35}
    numGSCmdV2Categs = 36
    
    
     #read split from files and all files in folders
    testWAVs = pd.read_csv(basePath+'/train/testing_list.txt', sep=" ", header=None)[0].tolist()
    valWAVs  = pd.read_csv(basePath+'/train/validation_list.txt', sep=" ", header=None)[0].tolist()
    

    testWAVs = [os.path.join(basePath+'/train/', f + '.npy') for f in testWAVs if f.endswith('.wav')]
    valWAVs  = [os.path.join(basePath+'/train/', f + '.npy') for f in valWAVs if f.endswith('.wav')]
    allWAVs  = []
    for root, dirs, files in os.walk(basePath+'/train/'):
        allWAVs += [root+'/'+ f for f in files if f.endswith('.wav.npy')]
    trainWAVs = list( set(allWAVs)-set(valWAVs)-set(testWAVs) )
    
    #get categories
    trainWAVlabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
    #print(trainWAVlabels)
    #build dictionaries
    trainWAVlabelsDict    = dict(zip(trainWAVs, trainWAVlabels))
    #print(trainWAVlabelsDict)
    #info dictionary
    trainInfo = {'files' : trainWAVs, 'labels' : trainWAVlabelsDict}
    gscInfo = {'train' : trainInfo}
    
    print('Done preparing Google Speech commands dataset version ')
    
    return gscInfo, numGSCmdV2Categs

def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    print(categ)
    return catDict.get(categ,0)

numberDirsDict = {1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine'}
def _getDyFileCategory(file, fileDict):
    categ = os.path.basename(os.path.dirname(file))
    foldername = None
    if len(categ) > 2 :
        foldername = categ[2:]
    else:
        foldername = numberDirsDict.get(int(categ))
        
    return fileDict.get(foldername)
        

def preparedyspeechcmd():
    DYSCmdCategs = {'unknown' : 0, 'one': 1, 'two' : 2, 'three' : 3, 'four' : 4, 'five' : 5, 'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9, 'close' : 10, 'up' : 11,
                    'down' : 12, 'previous' : 13, 'next' : 14, 'in' : 15, 'out' : 16,'left' : 17, 'right' : 18, 'home' : 19}
    numDYCmdCategs = 20
    _baseDir = '../Linzy/linzy_command/'
    firstLevelDirs = getNextLevelDirs(_baseDir)
    trainfiles = list()
    for folder in firstLevelDirs:
        d = os.path.join(_baseDir,folder)
        #print(d)
        files = [os.path.join(d,f+'.npy') for f in next(os.walk(d))[2] if f.endswith('.wav')]
        #print(files)
        #trainfiles.append([os.path.join(d,f+'.npy') for f in next(os.walk(d))[2] if f.endswith('.wav')])
        trainfiles += files
        
    #print(trainfiles)
    trainFilesLabels    = [_getDyFileCategory(f, DYSCmdCategs) for f in trainfiles]
    trainWAVlabelsDict    = dict(zip(trainfiles, trainFilesLabels))
    print(trainWAVlabelsDict)
    
    trainInfo = {'files': trainfiles, 'labels' : trainWAVlabelsDict}
    gscInfo = {'train' : trainInfo}
    
    return gscInfo, numDYCmdCategs
    
def getNextLevelDirs(root):
     return next(os.walk(root))[1]

sr=16000 #we know this one for google audios
iLen = 16000
batchSize = 32#64
#preparegooglespeechcmd()
_dscInfo, _numofCategs = preparedyspeechcmd()


def all_training_data_generation(list_of_files, _labels):
        
        X = np.empty((len(list_of_files),iLen)) #64 files, each file is 16000 long
        y = np.empty((len(list_of_files)),dtype=int)
        
        #Start to generate the training data
        for i, _f in enumerate(list_of_files):
            #print("current _f is : ",_f)
            #load npy file
            curX = np.load(_f)
            
            #check equal,smaller, or bigger
            #and truncate or padding
            #curX could be bigger or smaller than self.dim
            if curX.shape[0] == iLen:
                X[i] = curX
                #print('Same dim')
            elif curX.shape[0] > iLen: #bigger
                #we can choose any position in curX-self.dim
                randPos = np.random.randint(curX.shape[0]-iLen) 
                X[i] = curX[randPos:randPos+iLen]
                #print('File dim bigger')
            else: #smaller
                randPos = np.random.randint(iLen-curX.shape[0])
                X[i,randPos:randPos+curX.shape[0]] = curX
                #print('File dim smaller')
        # Store class
            y[i] = _labels[_f]

        return X, y

# generating all training data
x_train, y_train = all_training_data_generation(_dscInfo['train']['files'],_dscInfo['train']['labels'])
for i in range(len(x_train)) :
    print("x_train : {}, y_train : {}".format(x_train[i],y_train[i]))

#trainingData = DyTrainingDataGenerator.DySpeechGen(_dscInfo['train']['files'],_dscInfo['train']['labels'])
def testDataenerator(files,lbls):
    #_dsc['train']['files'],_dsc['train']['labels']
    print("_dscInfo'train']['files'] : ",_dscInfo['train']['files'])
    trainingData = DyTrainingDataGenerator.DySpeechGen(files,lbls)
    _audios, _classes = trainingData.__getitem__(0)
    print(trainingData.__len__())
    return _audios, _classes

#Start to Build the mfcc layer
'''
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

melspecModel = Sequential()

melspecModel.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                         padding='same', sr=sr, n_mels=80,
                         fmin=40.0, fmax=sr/2, power_melgram=1.0,
                         return_decibel_melgram=True, trainable_fb=False,
                         trainable_kernel=False,
                         name='mel_stft'
) )

melspecModel.add(Normalization2D(int_axis=0))
melspecModel.summary()
'''


# # Models
# 
# Create Keras models to see if the generators are working properly

from keras.models import model_from_json
from keras.models import Sequential
from keras.models import Model, load_model,save_model
from keras.layers import Input, Activation, Concatenate, Permute, Reshape, Flatten, Lambda, Dot, Softmax
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense#, LSTM, Bidirectional
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from kapre.time_frequency import Melspectrogram, Spectrogram
#from keras_tqdm import TQDMNotebookCallback


# In[23]:


#self-attention LSTM
#model = SpeechModels.AttRNNSpeechModel(_numofCategs, samplingrate = sr, inputLength = iLen)
#model = SpeechModels.ConvSpeechModel(_numofCategs, samplingrate = sr, inputLength = iLen)
model = SpeechModels.SimpleDNN(_numofCategs,inputLength = iLen)

sgd = SGD(lr=0.0000005, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer='sgd', loss = ['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy']) 
model.compile(optimizer='sgd', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
#model.compile(optimizer='adam', loss=['categorical_crossentropy'], metrics=['categorical_crossentropy'])
model.summary()


import math
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,  
            math.floor((1+epoch)/epochs_drop))
    
    if (lrate < 4e-5):
        lrate = 4e-5
      
    print('Changing learning rate to {}'.format(lrate))
    return lrate
lrate = LearningRateScheduler(step_decay)


#earlystopper = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10, verbose=1)
#checkpointer = ModelCheckpoint('AttRNN_model.h5', monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True)
#checkpointer = ModelCheckpoint('Conv_model.h5', monitor='val_sparse_categorical_accuracy', verbose=0, save_best_only=True)

#results = model.fit_generator(trainingData, validation_data = None, epochs = 40, use_multiprocessing=False, workers=1,
                    #callbacks=[earlystopper, checkpointer, lrate])
#result = model.fit(x_train,y_train, epochs = 20, batch_size = 64)
#callbacks_list = [checkpointer]
#result = model.fit(x_train,y_train, epochs = 20, batch_size = 64, callbacks=[TQDMNotebookCallback])
#result = model.fit(x_train,y_train, epochs = 20, batch_size = 64, callbacks=[lrate])
result = model.fit(x_train,y_train, epochs = 1000, batch_size = 32)

#model_json = model.to_json()
#with open("modelJSON.json", "w") as json_file:
 #   print("Serializing the model to json string.....")
  #  json_file.write(model_json)

print("Saving Model.........")
#model.save('AttRNN_model.h5')
model.save(os.path.join('.',DysTrainedModelPath,'DNN_model.h5'))
print("Print the history:\n",result)
