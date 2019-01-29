
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


# In[2]:
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#np.set_printoptions(threshold=np.nan)
#Load the CNN model
modelRootDict = "TrainedModels"
testRootDict = "../Linzy/Testing"
_cnn_model_path = os.path.join(".",modelRootDict,"Conv_model.h5") 
#_rnn_model_path = os.path.join(".",modelRootDict,"AttRNN_model.h5")
_dnn_model_path = os.path.join(".",modelRootDict,"DNN_model.h5") #DNN_model
_custom_objects={'Melspectrogram':Melspectrogram(),'Normalization2D':Normalization2D(int_axis=0)}
global theCNNModel
def __load_model(modelPath):
    theModel = load_model(modelPath,_custom_objects)
    return theModel

#DYSCmdCategsDigit = {0 : 'unknown', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 10:'close', 11:'up',
 #                   12:'down', 13:'previous', 14:'next', 15:'in', 16:'out', 17:'left', 18:'right', 19:'home'}

DYSCmdCategsDigit = {0:'one', 1:'two', 2:'three', 3:'four', 4:'five', 5:'six', 6:'seven', 7:'eight', 8:'nine', 9:'close', 10:'up',
                    11:'down', 12:'previous', 13:'next', 14:'in', 15:'out', 16:'left', 17:'right', 18:'home'}
# In[3]:


def _read_test_wav(wav_file):
    '''
    read the wav file and convert wav to numpy array.
    '''
    #testfile = os.path.join("..",'Linzy', testRootDict, wav_file)
    #print("reading the test wav file:",testfile)
    print("reading the test wav file:",wav_file)
    return librosa.load(wav_file)
    
'''
def _read_test_npy(npy_file):
        curX = np.load(_f)
            
        #check equal,smaller, or bigger
        #and truncate or padding
        #curX could be bigger or smaller than self.dim
        if curX.shape[0] == iLen:
                return curX
                #print('Same dim')
        elif curX.shape[0] > iLen: #bigger#we can choose any position in curX-self.dim
                randPos = np.random.randint(curX.shape[0]-iLen) 
                curX = curX[randPos:randPos+iLen]
                #print('File dim bigger')
        else: #smaller
                randPos = np.random.randint(iLen-curX.shape[0])
                curX[i,randPos:randPos+curX.shape[0]] = curX
        return curX
'''
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


def _adjustNPYShape(rawArray):
    X = np.empty(16000)
    len_of_rawArray = len(rawArray)
    print("length of rawArray is : ",len_of_rawArray)
    if len_of_rawArray == 16000:
            X = rawArray
            #print('Same dim')
    elif len_of_rawArray > 16000: #bigger
            #we can choose any position in curX-self.dim
            randPos = np.random.randint(len_of_rawArray-16000) 
            X = rawArray[randPos:randPos+16000]
            #print('File dim bigger')
    else: #smaller
            randPos = np.random.randint(16000-len_of_rawArray)
            print("smaller part - randPos : ",randPos)
            print("rawArray[0] : ",rawArray[0])
            X[0:len_of_rawArray] = rawArray
            #print('File dim smaller')
    return X


def main2():
    #print("Loading the Model...........")
    #__theDNNModel = __load_model(_dnn_model_path)
    print("Reading the test npy file.........")
    #y_npy = np.load("../linzyCut/test/LinZY03_13_6_previous.wav.npy")
    y_wav = _read_test_wav("../linzyCut/test/LinZY03_05_6_five.wav")
    print("y_npy's class name: ",y_npy.__class__.__name__)
    print("y_wav's class name: ",y_wav.__class__.__name__)
    print("y_npy's shape:",y_npy.shape)
    print("y_wav length ",len(y_wav))
    print("y_wav[0] type: ",y_wav[0].__class__.__name__)
    print("length of y_wav[0] : ",len(y_wav[0]))

    _



def cnnmain():
    print("Loading the Model...........")
    __theCNNModel = __load_model(_cnn_model_path)
    #__theDNNModel = __load_model(_dnn_model_path)
    #__theRNNModel = __load_model(_rnn_model_path)
    print("Reading the test wav file.........")
    
    raw_y, sr = librosa.load("../linzyCut/test/LinZY03_19_6_home.wav",sr=16000)#_read_test_wav("../linzyCut/test/LinZY03_01_6_one.wav")
    len_of_rawy = len(raw_y)
    #y_npy = np.load("../linzyCut/test/LinZY03_13_6_previous.wav.npy")
    y = _adjustNPYShape(raw_y)#np.transpose(_adjustShape(y))
    #print("has values part is {} : ",y[0:len_of_rawy])
    #__theCNNModel.summary()
    __theCNNModel.summary()
    print("type of y : ", y.__class__.__name__)
    print("shape of y is ", y.shape)
    input_y = y.reshape(1,16000)
    #print("content of input_y : ", input_y)
    #print("length is input_y: ",len(input_y))
    #print("shape's input_y is ",input_y.shape)
    y_result = __theCNNModel.predict(input_y)
    #y_result = __theDNNModel.predict(y)
    #y_result = __theRNNModel.predict(input_y)
    lenOfY = 19
    y_result_list = list(y_result[0])
    print(max(y_result_list))
    print("The right answer is home")
    print("====================================")
    for idx in range(19):
            print("y_result {0} element is: {1:6f} ==== {2}".format(idx, y_result[0,idx], DYSCmdCategsDigit.get(idx)))
    print("the shape result is : ",y_result.shape)
    #print("The predicting result is:",y_result)
    

def dnnmain():
    print("Loading the Model...........")
    __theCNNModel = __load_model(_cnn_model_path)
    #__theDNNModel = __load_model(_dnn_model_path)
    #__theRNNModel = __load_model(_rnn_model_path)
    print("Reading the test wav file.........")
    raw_y, sr = librosa.load("../linzyCut/test/LinZY03_04_6_four.wav",sr=16000)#_read_test_wav("../linzyCut/test/LinZY03_01_6_one.wav")
    len_of_rawy = len(raw_y)
    #y_npy = np.load("../linzyCut/test/LinZY03_13_6_previous.wav.npy")
    y = _adjustNPYShape(raw_y)#np.transpose(_adjustShape(y))
    #print("has values part is {} : ",y[0:len_of_rawy])
    #__theCNNModel.summary()
    __theCNNModel.summary()
    print("type of y : ", y.__class__.__name__)
    print("shape of y is ", y.shape)
    input_y = y.reshape(1,16000)
    #print("content of input_y : ", input_y)
    #print("length is input_y: ",len(input_y))
    #print("shape's input_y is ",input_y.shape)
    y_result = __theCNNModel.predict(input_y)
    #y_result = __theDNNModel.predict(y)
    #y_result = __theRNNModel.predict(input_y)
    lenOfY = 19
    y_result_list = list(y_result[0])
    print(max(y_result_list))
    print("====================================")
    for idx in range(19):
            print("y_result {0} element is: {1:6f} ==== {2}".format(idx, y_result[0,idx], DYSCmdCategsDigit.get(idx)))
    print("the shape result is : ",y_result.shape)
    #print("The predicting result is:",y_result)


# In[ ]:


if __name__ == "__main__":
    #npary, sample_rate = _read_test_wav("Ho1226_8894_0.wav")
    #print("NumPy Arrary of testing file is {0} and sample rate is {1}".format(npary, sample_rate))
    #print("Loading the Model...........")
    #__load_model(_cnn_model_path)
    cnnmain()

