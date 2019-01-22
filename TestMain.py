
# coding: utf-8

# ## The Main file to test the models

# In[1]:


import os
from keras.models import Model, load_model
#import audioUtils
import numpy as np
import librosa
import SpeechModels
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
from keras.layers import Add, Dropout, BatchNormalization, Conv2D, Reshape, MaxPooling2D, Dense, LSTM, Bidirectional

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# In[2]:


#Load the CNN model
modelRootDict = "TrainedModels"
testRootDict = "Testing"
_cnn_model_path = os.path.join(".",modelRootDict,"model-Conv2.h5")
_rnn_model_path = os.path.join(".",modelRootDict,"AttRNN_model.h5")
_custom_objects={'Melspectrogram':Melspectrogram(),'Normalization2D':Normalization2D(int_axis=0)}
global theCNNModel

#DYSCmdCategs = {'unknown' : 0, 'one': 1, 'two' : 2, 'three' : 3, 'four' : 4, 'five' : 5, 'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9, 'close' : 10, 'up' : 11,
 #                   'down' : 12, 'previous' : 13, 'next' : 14, 'in' : 15, 'out' : 16,'left' : 17, 'right' : 18, 'home' : 19}

DYSCmdCategsDigit = {0 : 'unknown', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'close', 11:'up',
                    12:'down', 13:'previous', 14:'next', 15:'in', 16:'out', 17:'left', 18:'right', 19:'home'}


def __load_model(modelPath):
    theModel = load_model(modelPath,_custom_objects)
    return theModel


# In[3]:


def _read_test_wav(wav_file):
    '''
    read the wav file and convert wav to numpy array.
    '''
    testfile = os.path.join(".", testRootDict, wav_file)
    print("reading the test wav file:",testfile)
    return librosa.load(testfile,16000)
    


# In[4]:


def _adjustShape(rawArray):
    X = np.empty((1,16000))
    if rawArray[0] == 16000:
            X = rawArray
            #print('Same dim')
    elif rawArray.shape[0] > 16000: #bigger
            #we can choose any position in curX-self.dim
            randPos = np.random.randint(rawArray.shape[0]-16000) 
            X = rawArray[randPos:randPos+16000]
            #print('File dim bigger')
    else: #smaller
            randPos = np.random.randint(16000-rawArray.shape[0])
            X[i,randPos:randPos+rawArray.shape[0]] = rawArray
            #print('File dim smaller')
    return X


# In[ ]:

float_formatter = lambda x: "%1.6f" % x

def main():
    print("Loading the Model...........")
    #__theCNNModel = __load_model(_cnn_model_path)
    __theRNNModel = __load_model(_rnn_model_path)
    print("Reading the test wav file.........")
    y, sr = _read_test_wav("LinZY03_10_6_close.wav")
    y = _adjustShape(y)#np.transpose(_adjustShape(y))
    #__theCNNModel.summary()
    __theRNNModel.summary()
    input_y = y.reshape(1,16000)
    print(y.shape)
    #print("y's shape is {0} and {1}".format(y.shape, sr))
    #y_result = __theCNNModel.predict(input_y)
    y_result = __theRNNModel.predict(input_y)
    lenOfY = 20
    y_result_list = list(y_result[0])
    print(max(y_result_list))
    print("====================================")
    for idx in range(20):
            print("y_result {0} element is: {1:6f} ==== {2}".format(idx, y_result[0,idx], DYSCmdCategsDigit.get(idx)))
    #print("the test result is : ",y_result.shape)
    #print("The predicting result is:",y_result)
    


# In[ ]:


if __name__ == "__main__":
    #npary, sample_rate = _read_test_wav("Ho1226_8894_0.wav")
    #print("NumPy Arrary of testing file is {0} and sample rate is {1}".format(npary, sample_rate))
    #print("Loading the Model...........")
    #__load_model(_cnn_model_path)
    main()

