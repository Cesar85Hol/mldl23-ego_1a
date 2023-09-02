import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet34,alexnet
import torchvision
from torchvision import transforms
import numpy as np
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
# some_file.py
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './actionNet')
filepath=''
import visualize_spec
ann_I3D_ActionNet=pd.DataFrame()

#from actionNet import visualize_spec
#import librosa 
class AlexNet(torch.nn.Module): #Module is base class for all neural network modules.
  def __init__(self,n_classes=20) :
    super(AlexNet,self).__init__()
    self.layer1=nn.Sequential(
                nn.Conv2d(1,96,kernel_size=11,stride=4,padding=0),#1st question, which filter I gave?? 1 0 -1 x nrows always?
                nn.BatchNorm2d(96),#turn this off to see different results
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2))
    
    self.layer2=nn.Sequential(
                nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),#1st question, which filter I gave?? 1 0 -1 x nrows always?
                nn.BatchNorm2d(256),#turn this off to see different results
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2))
    self.layer3=nn.Sequential(
                nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),#1st question, which filter I gave?? 1 0 -1 x nrows always?
                nn.BatchNorm2d(384),#turn this off to see different results
                nn.ReLU())
    self.layer4=nn.Sequential(
                nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),#1st question, which filter I gave?? 1 0 -1 x nrows always?
                nn.BatchNorm2d(384),#turn this off to see different results
                nn.ReLU())
    self.layer5=nn.Sequential(
                nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),#1st question, which filter I gave?? 1 0 -1 x nrows always?
                nn.BatchNorm2d(256),#turn this off to see different results
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2))
    self.fc1= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096,2048),#nn.Linear(9216,4096),
        nn.ReLU())  
    self.fc2= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048,2048),
        nn.ReLU())
    self.fc3= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048,n_classes),
        nn.ReLU())

    #DO SOMETHING
  def forward(self,x): # define the sequence in which layers will process the image. This is defined inside the forward function
    
    out=self.layer1(x)
    out=self.layer2(out)
    out=self.layer3(out)
    out=self.layer4(out)
    out=self.layer5(out)
    #print(f"size of output before FC layers: {out.shape}")
    out = out.reshape(out.size(0), -1)#to make it 1 dimensional
    #print(f"size of output after FC layers: {out.shape}")
    out=self.fc1(out)
    out=self.fc2(out)
    out=self.fc3(out)
    #print(f"size of output before exiting forward: {out.shape}")
    return out
  
class custom_2DCNN_test(nn.Module):
    def __init__(self, nlayers, n_channels=1,n_classes=20) -> None:
        super().__init__()
        self.layers=nlayers
        self.channels=n_channels # ready in case we want to work with 2 channels separated (left and right)
        self.n_classes= n_classes
        self.layer1=nn.Sequential(
            nn.Conv2d(self.channels,32,kernel_size=2,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 49-3+2=48
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            # nn.AvgPool2d(kernel_size=2,stride=1),#13x9
            nn.AvgPool2d(2,2),
            nn.Dropout(0.2)
        )#params= 3*3
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 48
            nn.BatchNorm2d(64),#if dropout iimportant to be after normlization
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            nn.AvgPool2d(2,2),#12,8
            nn.Dropout(0.2)
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 112-5+2+1=55
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            nn.AvgPool2d(2,2),#6,4
            nn.Dropout(0.5)
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 112-5+2+1=55
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            nn.AvgPool2d(2,2),#6,4
            nn.Dropout(0.2)
        )
        self.layer5=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 112-5+2+1=55
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            #nn.AvgPool2d(2,2),#6,4
            nn.Dropout(0.5)
        )
        self.layer6=nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1), #ouput size: W-F+2P)/S+1: 112-5+2+1=55
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.1,inplace=True),
            nn.AvgPool2d(2,2),#3,2
            nn.Dropout(0.5)
        )

        self.fc1= nn.Sequential(
        nn.Linear(128*2*3,1024),#test 8 -  overfitting
        #nn.Linear(128*3*2,1024),#test 7 (avgpool 2,2)
        #nn.Linear(128*9*13,1024),#test1-6
        #nn.Linear(27*27*32,1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        # nn.LeakyReLU(0.1,inplace=True),
        nn.Dropout(0.5)
        )
        self.fc2=nn.Linear(1024,256)
        self.dropout=nn.Dropout(0.5)
        self.fc3=nn.Linear(256,self.n_classes)
        self.softmax=nn.Softmax(dim=0)
    
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        #out=self.layer4(out)
        #out=self.layer5(out)
        #out=self.layer6(out)
        #print(f"size of output before FC layers: {out.shape}")
        out = out.reshape(out.size(0), -1)#to make it 1 dimensional
        #print(f"size of output after FC layers: {out.shape}")
        out=self.fc1(out)
        out=self.fc2(out)
        out=self.dropout(out)
        out=self.fc3(out)
        #out=self.softmax(out)
        #print(f"size of output before exiting forward: {out.shape}")
        return out

class custom_2DCNN(nn.Module):
    def __init__(self, nlayers, n_channels=1,n_classes=20) -> None:
        super().__init__()
        self.layers=nlayers
        self.channels=n_channels # ready in case we want to work with 2 channels separated (left and right)
        self.n_classes= n_classes
        self.layer1=nn.Sequential(
            nn.Conv2d(self.channels,32,kernel_size=3,padding=1), #ouput size: (224-3+2*1)/1 + 1=> W2=224,H2=224
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2))#nn.MaxPool2d(kernel_size=3,stride=2))#from AlexNet #output size= W=(224-2)/2 +1=112
        self.layer2=nn.Sequential(
            nn.Conv2d(32,128,kernel_size=3,stride=1,padding=1), #ouput size: W=112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)) #output size W= 112-2)/2+1 = 56
        self.layer3=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1), #ouput size: 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)) #W,H= 56-2)/2+1=54/2+1=28 --> output finale = 28*28*256=200704 , 28*28*128=100352, 28*28*64=50176
        self.fc1= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(50176,4096),
        nn.ReLU())
        self.fc2= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096,self.n_classes),
        nn.ReLU())
        self.softmax=nn.Softmax()
    
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        #print(f"size of output before FC layers: {out.shape}")
        out = out.reshape(out.size(0), -1)#to make it 1 dimensional
        #print(f"size of output after FC layers: {out.shape}")
        out=self.fc1(out)
        out=self.fc2(out)
        out=self.softmax(out)
        #print(f"size of output before exiting forward: {out.shape}")
        return out

class EMG_Classifier(nn.Module):

    def __init__(self, input_len,hidden_layers,num_layers,num_classes,batch_size):
        super().__init__()
        self.hidden_layers=hidden_layers
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.num_layers=num_layers
        self.input_len=input_len
        # self.lstm1=nn.LSTM(input_size=100,hidden_size=self.hidden_layers, batch_first=True)#,works for time dimension (100 sample length)
        #self.lstm1=nn.LSTM(input_size=26,hidden_size=self.hidden_layers, batch_first=True)#,num_layers=1, batch_first=True) # working for spectrograms
        self.lstm1=nn.LSTM(self.input_len,self.hidden_layers, self.num_layers,batch_first=True)#,num_layers=1, batch_first=True)

        self.lstm2= nn.LSTM(self.hidden_layers,50)
        self.dropout1=nn.Dropout(0.2)
        # self.dense=nn.Linear(50,self.num_classes)
        self.dense=nn.Linear(416*50,self.num_classes)

        self.softmax=nn.Softmax()

    def forward(self, x):
        # h_t = torch.zeros(self.batch_size,1, self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        # c_t = torch.zeros(self.batch_size,1, self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        # h_t = torch.zeros(1,x.size(0), self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        # c_t = torch.zeros(1,x.size(0), self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        h_t = torch.zeros(self.num_layers,x.size(0), self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        c_t = torch.zeros(self.num_layers,x.size(0), self.hidden_layers, dtype=torch.float32, device=torch.device("mps"))
        # h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # x,_=self.lstm1(x.view(-1,self.batch_size,x.size(3)),(h_t,c_t))
        out,_=self.lstm1(x,(h_t,c_t))
        out,_=self.lstm2(out)
        out=self.dropout1(out)
        #maybe before appliying the linear module, some reshape it's needed
        out=out.reshape(self.batch_size,-1)
        #out=self.dense(out[:,-1,:])
        #x=self.dense(x[-1,:,:])
        out=self.dense(out)
        #out=self.softmax(out)
        return out

def import_data(arm=0):
    # Open the file.
    global filepath
    filepath = '../ActionNet_data/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5'
    h5_file = h5py.File(filepath, 'r')

    ####################################################
    # Example of reading sensor data: read Myo EMG data.
    ####################################################
    #print()
    #print('='*65)
    #print('Extracting EMG data from the HDF5 file')
    #print('='*65)
    if arm==0:
        device_name = 'myo-left'
    else:
        device_name= 'myo-right'
    stream_name = 'emg'
    # Get the data as an Nx8 matrix where each row is a timestamp and each column is an EMG channel.
    emg_data = h5_file[device_name][stream_name]['data']
    emg_data = np.array(emg_data)
    # Get the timestamps for each row as seconds since epoch.
    emg_time_s = h5_file[device_name][stream_name]['time_s']
    emg_time_s = np.squeeze(np.array(emg_time_s)) # squeeze (optional) converts from a list of single-element lists to a 1D list
    # Get the timestamps for each row as human-readable strings.
    emg_time_str = h5_file[device_name][stream_name]['time_str']
    emg_time_str = np.squeeze(np.array(emg_time_str)) # squeeze (optional) converts from a list of single-element lists to a 1D list

        #convert in tensor
        #get just EMG data
    
    ####################################################
    return emg_data,emg_time_s,emg_time_str

def transform_2Resnet(X, transform_toPil,transforms_train):
    
    #X=X.sum(axis=1) # since pil does not accept channels>4 --> summ all 16 chs
    #X=X.reshape(-1,4,X.size(2),X.size(3))
    #TRANSFORM IMAGE TO SIZE 224X224
    x_=[]
    for i in range(X.size(0)):
        #tmp=transform_toPil(X[i])#PIL needed to apply transforms
        #tmp=tmp.resize((224,224))
        #tmp=transform_toPil(torch.stack((X[i],X[i],X[i])))# 3 channels for original resnet
        #tmp=transforms_train(tmp)# check if works
        #tmp=torch.stack((tmp,tmp,tmp))#necessary to have 3 channels for the resnet
        #tmp=tmp.reshape(-1,tmp.size(2),tmp.size(3))
        #x_.append(tmp)
        x_.append(X[i])
    X=torch.stack(x_)
    return X

def train(model,loader,dataset,emg_dataL,emg_dataR,optimizer,loss_function,device,transform_toPil,transforms_train):
    model.train()
    samples=0.
    cumulative_loss=0.
    cumulative_accuracy=0.
    cumulative_accuracytop5=0.

    n_class_corr=[0 for i in range(20)]
    n_class_samples=[0 for i in range(20)]
    for batch_idx,(idx,y) in enumerate(loader):
        #trasformare ixd in X
        #_,X=get_values(dataset,idx,emg_time_s,emg_data)
        #X=get_values(dataset,idx,emg_dataL) #working with time --> 
        
        #X=torch.Tensor(emg_data[int(dataset.iloc[idx]['start_frame']):int(dataset.iloc[idx]['stop_frame'])])
        X=get_values_spectr(dataset,idx,emg_dataL,emg_dataR) #has to return the spectrgorams
        #X=transform_2Resnet(X,transform_toPil,transforms_train) #if applied transforms
        #X=X.reshape(-1,X.size(3),X.size(2))# X.size(3)->seq_len; X.size(2)->inpout_len
        X,y=X.to(device),y.to(device)
        outputs=model(X)
        loss=loss_function(outputs,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        samples += idx.shape[0]
        cumulative_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        #_, predicted = outputs.max(1)
        cumulative_accuracy += predicted.eq(y).sum().item()
        cumulative_accuracytop5+=get_accuracytop5(outputs,y)

        for i in range(X.size(0)):
                label=y[i]
                pred=predicted[i]
                if(label==pred):
                    n_class_corr[label]+=1
                n_class_samples[label]+=1
    # print('-'*40)
    # print('ONLY TRAINING')
    # for i in range(20):
    #     acc=100*n_class_corr[i]/n_class_samples[i]
    #     print(f'Accuracy of class {i} :{acc}')
    return cumulative_loss/samples, cumulative_accuracy/samples*100, cumulative_accuracytop5/samples*100

def test(model,loader,dataset,emg_dataL,emg_dataR,loss_function,device,transform_toPil,transforms_test):
    model.eval()
    samples=0.
    cumulative_loss=0.
    cumulative_accuracy=0.
    cumulative_accuracytop5=0.
    with torch.no_grad():
        
        n_class_corr=[0 for i in range(20)]
        n_class_samples=[0 for i in range(20)]
        for batch_idx,(idx,y) in enumerate(loader):
            #trasformare ixd in X
            #_,X=get_values(dataset,idx,emg_time_s,emg_data)
            #X=get_values(dataset,idx,emg_dataL) #working with time --> 
            
            #X=torch.Tensor(emg_data[int(dataset.iloc[idx]['start_frame']):int(dataset.iloc[idx]['stop_frame'])])
            X=get_values_spectr(dataset,idx,emg_dataL,emg_dataR) #has to return the spectrgorams
            #X=X.reshape(-1,X.size(3),X.size(2))# X.size(3)->seq_len; X.size(2)->inpout_len
            X=transform_2Resnet(X,transform_toPil,transforms_test)
            X,y=X.to(device),y.to(device)
            outputs=model(X)
            loss=loss_function(outputs,y)
            samples += idx.shape[0]
            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            #_, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(y).sum().item()
            cumulative_accuracytop5 += get_accuracytop5(outputs,y)

            for i in range(X.size(0)):
                label=y[i]
                pred=predicted[i]
                if(label==pred):
                    n_class_corr[label]+=1
                n_class_samples[label]+=1
    #print('-'*40)
    #print('ONLY TEST')
    for i in range(20):
        acc=100*n_class_corr[i]/n_class_samples[i]
       # print(f'Accuracy of class {i} :{acc}')
    return cumulative_loss/samples, cumulative_accuracy/samples*100, cumulative_accuracytop5/samples*100

def get_accuracytop5(predicted,GT):
    cumulAcc=0.
    sorted_=predicted.data.argsort(dim=1,descending=True)
    pred5=sorted_[:,:5]
    res=[]
    #cumulAcc += predicted.eq(GT).sum().item()
    for i in range(predicted.size(0)):# batch
        y=GT[i]
        res.append([1 if x==y else 0 for x in pred5[i]])
    res=np.array(res)
    top5=np.max(res,1)
    return sum(top5) 

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def transform_data(emg,time,single_ch=True):
    emg=emg.transpose()
    emg=abs(emg)
    emg_filt=butter_lowpass_filter(emg,5,160)
    minimum=emg_filt.min()
    maximum=emg_filt.max()
    emg_transf=[(ch-minimum)/(maximum-minimum) for ch in emg_filt]
    if single_ch:
        emg_transf=np.array(emg_transf).sum(axis=0)
    return np.transpose(emg_transf)

def create_dataset(emg_time_s_L,emg_time_s_R,t_strR):
    global filepath #= '../ActionNet_data/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5'
    h5_file = h5py.File(filepath, 'r')
    device_name = 'experiment-activities'
    stream_name = 'activities'

    # Get the timestamped label data.
    # As described in the HDF5 metadata, each row has entries for ['Activity', 'Start/Stop', 'Valid', 'Notes'].
    activity_datas = h5_file[device_name][stream_name]['data']
    activity_times_s = h5_file[device_name][stream_name]['time_s']
    activity_times_s = np.squeeze(np.array(activity_times_s))  # squeeze (optional) converts from a list of single-element lists to a 1D list
    # Convert to strings for convenience.
    activity_datas = [[x.decode('utf-8') for x in datas] for datas in activity_datas]

    # Combine start/stop rows to single activity entries with start/stop times.
    #   Each row is either the start or stop of the label.
    #   The notes and ratings fields are the same for the start/stop rows of the label, so only need to check one.
    exclude_bad_labels = True # some activities may have been marked as 'Bad' or 'Maybe' by the experimenter; submitted notes with the activity typically give more information
    activities_labels = []
    activities_start_times_s = []
    activities_end_times_s = []
    activities_ratings = []
    activities_notes = []
    for (row_index, time_s) in enumerate(activity_times_s):
        label    = activity_datas[row_index][0]
        is_start = activity_datas[row_index][1] == 'Start'
        is_stop  = activity_datas[row_index][1] == 'Stop'
        rating   = activity_datas[row_index][2]
        notes    = activity_datas[row_index][3]
        if exclude_bad_labels and rating in ['Bad', 'Maybe']:
            continue
        # Record the start of a new activity.
        if is_start:
            activities_labels.append(label)
            activities_start_times_s.append(time_s)
            activities_ratings.append(rating)
            activities_notes.append(notes)
        # Record the end of the previous activity.
        if is_stop:
            activities_end_times_s.append(time_s)

    # Get EMG data for the first instance of the second label.
    u_targets=pd.unique(activities_labels)
    dataset=pd.DataFrame()
    tmp=[]
   
    for target_label in u_targets:
        #target_label = activities_labels[1]
        #target_label_instance = 0
        # Find the start/end times associated with all instances of this label.
        label_start_times_s = [t for (i, t) in enumerate(activities_start_times_s) if activities_labels[i] == target_label]
        label_end_times_s = [t for (i, t) in enumerate(activities_end_times_s) if activities_labels[i] == target_label]
        #tmp=[(start,end) for start,end in zip(label_start_times_s,label_end_times_s)] #working for 1 sample per action
        tmpLeft=[get_indexes(start,end,emg_time_s_L) for start,end in zip(label_start_times_s,label_end_times_s)] #splits samples in 100 subsamples length
        tmpRight=[get_indexes(start,end,emg_time_s_R) for start,end in zip(label_start_times_s,label_end_times_s)]
        dfL=pd.DataFrame()
        dfR=pd.DataFrame()
        for row in tmpLeft:
            dfL=pd.concat([dfL,pd.DataFrame(row).transpose()])
        for row in tmpRight:
            dfR=pd.concat([dfR,pd.DataFrame(row).transpose()])
        
        dfR.reset_index(drop=True,inplace=True)
        dfL.reset_index(drop=True,inplace=True)
        df=pd.concat([dfL,dfR],axis=1)
        df=pd.concat([df,pd.DataFrame([target_label]*df.shape[0])],axis=1)
        df.dropna(inplace=True)# since considering both channels together drop samples where activities are not considered, if want to consider just 1 arm consider to comment this line
        dataset=pd.concat([dataset,df])

    dataset.reset_index(drop=True,inplace=True)
    dataset.columns=['start_frame_L','stop_frame_L','start_frame_R','stop_frame_R','narration']#'label']
    dataset['item']=dataset.index
    #transform labels in integers
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(dataset['narration'])#'label'])
    dataset['label']=targets

    dataset['verb']=[dataset['narration'][i].split()[0] for i in range(dataset['item'].size)]#dataset_origin['narration'][0] #get first word in the string is the verb
    le2 = preprocessing.LabelEncoder()
    targets = le2.fit_transform(dataset['verb'])
    dataset['verb_class']=targets
    create_annotation_ActionNet_I3D(dataset,t_strR)
    return dataset,le #le necessary to transform ints to labels

def create_annotation_ActionNet_I3D(dataset_origin,time4video):
    global ann_I3D_ActionNet
    ann_I3D_ActionNet['uid']=dataset_origin['item']
    ann_I3D_ActionNet['video_id']=["" for i in range(ann_I3D_ActionNet['uid'].size)]#for correctly loading with ek-loaders file
    ann_I3D_ActionNet['start_frame']=getVideoFrame(dataset_origin,time4video,True)
    ann_I3D_ActionNet['stop_frame']=getVideoFrame(dataset_origin,time4video,False)
    ann_I3D_ActionNet['participant_id']=['S4' for i in range(ann_I3D_ActionNet['uid'].size)]
    ann_I3D_ActionNet['narration']=dataset_origin['narration']
    ann_I3D_ActionNet['narration_class']=dataset_origin['label']
    ann_I3D_ActionNet['verb']=dataset_origin['verb']
    ann_I3D_ActionNet['verb_class']=dataset_origin['verb_class']
    
    #code to write test/train annotation files
    '''
    X_train,X_test,_,_=train_test_split(ann_I3D_ActionNet['uid'],ann_I3D_ActionNet['narration_class'], test_size=0.2,random_state=1234)
    ActionNet_train=ann_I3D_ActionNet.iloc[X_train]
    ActionNet_test=ann_I3D_ActionNet.iloc[X_test]
    ActionNet_test.to_pickle("./actionNet/S04_test.pkl")
    ActionNet_train.to_pickle("./actionNet/S04_train.pkl")
    '''
    return

def getVideoFrame(dataset,time4video,start):
    column=''
    startvideo=16*3600+38*60+43
    frame_list=[]
    if start:
        column="start_frame_R" 
    else:
        column="stop_frame_R" 
    for i in range(dataset.shape[0]):
        start=time4video[int(dataset[column][i])]
        s=start.decode('utf-8').split(" ")[1]
        time=int(s.split(':')[0])*3600+int(s.split(':')[1])*60+float(s.split(':')[2])
        delta_t=time-startvideo
        frame=round(delta_t*30) # to understand the frame multiply time*framerate
        frame_list.append(frame)
    return frame_list

def get_values(dataset,idx,emg_data,emg_data2=0): #old for getting values in time for a single arm
    '''label_start_time_s=dataset.iloc[index]['start_time_s']
    label_end_time_s=dataset.iloc[index]['stop_time_s']

    # Segment the data!
    emg_indexes_forLabel = np.where((emg_time_s >= label_start_time_s.values) & (emg_time_s <= label_end_time_s.values))[0]
    emg_data_forLabel = emg_data[emg_indexes_forLabel]
    #emg_data_forLabel = emg_data[emg_indexes_forLabel, :]

    emg_time_s_forLabel = emg_time_s[emg_indexes_forLabel]
    #emg_time_str_forLabel = emg_time_str[emg_indexes_forLabel]

    return torch.Tensor(emg_data_forLabel)
    '''
    x=[]
    for index in idx:
        start=int(dataset.iloc[int(index)]['start_frame_L'])
        stop=int(dataset.iloc[int(index)]['stop_frame_L'])
        tmp= emg_data[start:stop]
        x.append(tmp)
    return torch.tensor(x,dtype=torch.float32)

def Spectrogram(wave, wav_name, fs, plot=False):
    '''

    Parameters
    ----------
    wave : wav file, SINGLE AUDIO FILE.
    wav_name : string, AUDIO FILE NAME (necessary when plot is True)
    fs : int, SAMPLE FREQUENCY .
    plot : bool. The default is False.

    Returns
    -------
    f : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    Sxx : TYPE
        DESCRIPTION.

    '''

    # SETTING SUGGESTED VALUES FOR SPECTROGRAM BOOK MACHINE LEARNING..12.3.2 The Speech Front End
    deltat = int(0.02*fs)  # overlapping of 20ms
    width = int(0.03*fs)  # window of 30 ms book
    w = signal.windows.hamming(width)
    f, t, Sxx = signal.spectrogram(wave, fs, window=w, noverlap=deltat)
    if plot:
        fig3, ax3 = plt.subplots()
        im = ax3.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        fig3.colorbar(im, ax=ax3)
        ax3.set_ylabel('Frequency [Hz]')
        # ax3.set( yscale="log")
        ax3.set_xlabel('Time [sec]')
        ax3.set_title(wav_name)
        return fig3, ax3
    else:
        return f, t, Sxx

def get_values_spectr(dataset,idx,emg_data,emg_data2):# get spectrograms and build an image with 16 channels
    
    x=[]
    for index in idx:
        startL=int(dataset.iloc[int(index)]['start_frame_L'])
        stopL=int(dataset.iloc[int(index)]['stop_frame_L'])
        tmp= emg_data[startL:stopL]
        startR=int(dataset.iloc[int(index)]['start_frame_R'])
        stopR=int(dataset.iloc[int(index)]['stop_frame_R'])
        tmpR= emg_data2[startR:stopR]
        tmp_spec=visualize_spec.compute_spectrogram(emg_data[startL:stopL],"LSTM")
        tmp_spec2=torch.cat((tmp_spec,visualize_spec.compute_spectrogram(emg_data2[startR:stopR],"LSTM")),0)

        x.append(tmp_spec2)
        #x.append(tmp_specR)
    #return torch.tensor(x,dtype=torch.float32)
    return (torch.stack(x)).to(torch.float32)

def get_indexes(start,stop,emg_time_s):
    label_start_time_s=start
    label_end_time_s=stop
    # Segment the data!
    emg_indexes_forLabel = np.where((emg_time_s >= label_start_time_s) & (emg_time_s <= label_end_time_s))[0]
    start=[]
    stop=[]
    for i in range(emg_indexes_forLabel[0],emg_indexes_forLabel[-1]-100,100):
        start.append(i)
        stop.append(i+100)    
    return list([start,stop])

def get_info_from_pkls():
    import pickle
    train_path="./train_val/S04_train.pkl" 
    test_path="./train_val/S04_test.pkl" 

    file_tr=open(train_path,'rb')
    file_te=open(test_path,'rb')
    feat_tr=pickle.load(file_tr)
    feat_te=pickle.load(file_te)

    #return feat_tr['uid'],feat_te['uid'],feat_tr['verb_class'],feat_te['verb_class']
    return feat_tr['uid'],feat_te['uid'],feat_tr['narration_class'],feat_te['narration_class']

def main():
    sequence_len=26 #params for output images of spectrogram [2,49]
    input_len=17
    hidden_size=5
    num_layers=2
    num_classes=20 #do not consider base class 0
    device = torch.device("mps" if torch.has_mps else "cpu")#conv3D is still not working on mps, leave cpu
    batch_size=16
    learning_rate=0.001 #0.001
    epochs=150
    momentum=0.9
    weight_decay=0.000001 #0.000001
    emg_L, time_s_L,time_str_L= import_data(0)
    emg_R,time_s_R,time_str_R= import_data(1)
    stifness_L=transform_data(emg_L,time_s_L,False)
    stifness_R=transform_data(emg_R,time_s_R,False)
    dataset_info,le=create_dataset(time_s_L,time_s_R,time_str_R)#first 2 to get emg data, 3rd to get correct frames from the video
    #dataset_info,le=create_dataset(time_s_L,emg_L,time_str_L)
    ###downsampling
    # index=dataset_info[dataset_info['label']==6]
    # new_index=index.sample(350)
    # dataset_info=dataset_info[717:]#cut previous data
    # dataset_info=pd.concat([new_index,dataset_info])
    #################
    #X_train,X_test,y_train,y_test=train_test_split(dataset_info['item'],dataset_info['label'], test_size=0.2,random_state=1234)
    X_train,X_test,y_train,y_test=get_info_from_pkls()
    loader_train= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_train)), torch.tensor(np.array(y_train))), shuffle=True, batch_size=batch_size,drop_last=True)
    loader_test= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_test)), torch.tensor(np.array(y_test))), shuffle=False, batch_size=batch_size,drop_last=True)

    loss_function=get_loss_function()
    modelEMG=EMG_Classifier(input_len,hidden_size,num_layers,num_classes,batch_size) #check correttezza implementazione rete
    #Resnet 18 implementation & modification
    model_Resnet18= resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1) #use pretrained 2dCNN
    model_Resnet18.conv1=nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)#so first layer accepts 1 channel (sum of spectrogram)
    num_features = model_Resnet18.fc.in_features     #extract fc layers features
    model_Resnet18.fc = nn.Linear(num_features, num_classes) #(num_of_class == 20)
    #########################################
    
    #CUSTOM 2D CNN
    #model2dCNN=custom_2DCNN(1,n_classes=num_classes)#to understand if batch_size has to be passed
    model_test=custom_2DCNN_test(1,n_channels=16,n_classes=20)#2 with other spectrogram
    #modelAN=AlexNet()
    ########################################

    transform_toPil = transforms.ToPILImage()
    transforms_train = transforms.Compose([
        #transforms.Resize((224, 224)),   #must same as here - upsample provare . cambia la resoluzione originale
        #transforms.RandomCrop(224,224),
        #transforms.RandomHorizontalFlip(), # data augmentation
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5)) # normalization
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization

    ])
    transforms_test = transforms.Compose([
        #transforms.Resize((224, 224)),   #must same as here
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5))
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #select the model
    #model=model_Resnet18
    #model=model2dCNN
    model=model_test
    model.to(device)
    ##############################
    optimizer=get_optimizer(model,learning_rate,weight_decay,momentum,0)#0:sgd - 1:adam
    print(model)
    val_loss_list = []
    val_accuracy_list = []
    train_loss_list = []
    train_accuracy_list = []
    for epoch in range(epochs):
        
        train_loss,train_accuracy,train_accuracy_top5=train(model,loader_train,dataset_info,stifness_L,stifness_R,optimizer,loss_function,device,transform_toPil,transforms_train)#inserire il data loader nel processo, seguire esempi per il training
        print("-"*40)
        print('Epoch: {:d}'.format(epoch+1))
        print('\t Training loss {:.5f}, Training accuracy top1 {:.2f}, top5 {:.2f}'.format(train_loss,train_accuracy,train_accuracy_top5))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
        # val_accuracy))
        print("-"*40)
        
        
        test_loss,test_accuracy, test_accuracy_top5=test(model,loader_test,dataset_info,stifness_L,stifness_R,loss_function,device,transform_toPil,transforms_test)#
        #test()#ricordarsi di usare torch.no_grad e poi procedere al test e incrociare le dita :D
        val_loss_list.append(test_loss)
        val_accuracy_list.append(test_accuracy)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        # print("After training:")
        # train_loss, train_accuracy = test(model, train_loader, loss_function,device)
        # val_loss, val_accuracy = test(model, val_loader, loss_function,device)
        # test_loss, test_accuracy = test(net, test_loader, loss_function,device)
        # print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
        # train_accuracy))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
        # val_accuracy))
        print('\t Test loss {:.5f}, Test accuracy top1 {:.2f}, top5 {:.2f}'.format(test_loss, test_accuracy,test_accuracy_top5))
        print('-'*40)
        #plot_accuracy(train_accuracy_list,val_accuracy_list,1,0,"test:softmax")
        #plot_accuracy(train_loss_list,val_loss_list,0,0,"test: softamx")

    nametosave="test_2DCNN-7_from_annotation"   
    save_path = nametosave+'.pth'
    torch.save(model.state_dict(), save_path)
    # plot_accuracy(train_accuracy_list,val_accuracy_list,"Resnet18-EMG_classification-8channelsconcat_summedperarm")
    plot_accuracy(train_accuracy_list,val_accuracy_list,1,1,nametosave)
    plot_accuracy(train_loss_list,val_loss_list,0,1,nametosave+"-loss")

    return

def get_loss_function():
    loss_function= torch.nn.CrossEntropyLoss()
    return loss_function

def plot_accuracy(train_accuracy,val_accuracy,type,save_,model_name=""):#type0acc,type1loss
    import seaborn as sns
    sns.set_theme()
    palette = sns.color_palette('pastel')       
    fig,ax = plt.subplots()#(figsize=(10,10))
    sns.lineplot(x=list(range(1,len(train_accuracy)+1)), y=train_accuracy,marker='o', label='train',ax=ax)# , col="Frames",height=4, aspect=0.6,)
    sns.lineplot(x=list(range(1,len(train_accuracy)+1)), y=val_accuracy,marker='o', label='test',ax=ax)# , col="Frames",height=4, aspect=0.6,)
    ax.set_xlabel('num_epochs')
    if (type):
        ax.set_title(f"Train-Validation Accuracy - model: {model_name}")
        ax.set_ylabel('accuracy')
    else:
        ax.set_title(f"Loss - model: {model_name}")
        ax.set_ylabel('Loss')
    ax.legend()
    print('train final accuracy: {:.2f} '.format(np.mean(train_accuracy[-10:])))
    print('test final accuracy: {:.2f} '.format(np.mean(val_accuracy[-10:])))


    if save_:
        fig.savefig(model_name+".pdf")
        dict_model={'epochs':list(range(1,len(train_accuracy)+1)),'train_accuracy':train_accuracy,'test_accuracy':val_accuracy,'model':model_name}
        pd.DataFrame(dict_model).to_pickle(model_name+".pkl")

def get_optimizer(net,lr,wd,momentum,kind=0):
    if kind ==0:
        optimizer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd,momentum=momentum)
    else:
        optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    return optimizer

main()
