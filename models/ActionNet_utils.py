'''
1. download **.hdf5 file in ActionNet_data folder (same level of ek_data)
2. copy ActionNet_utils.py on the "models" folder
3. copy visualize_spec.py on "actionNet" folder
4. run the script ActionNet_utils : I created a main function so you can see the dataset created. it's in the form:
| start_frame_L | stop_frame_L | start_frame_R | stop_frame_R | Label | item |
rows are records
- These frames are used to find the related times on the emg_time_L/R variables
To create the annotation files for rgb extraction on actionNet we have to match the times between emg_time_L/R (EMG). See create_dataset & get_indexes functions
and the frames of the video. 
- Video frames are saved in the folders, from the start there are 5 frames each second (video framrate was 25Hz)
'''
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

def create_dataset(emg_time_s_L,emg_time_s_R):

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
    dataset.columns=['start_frame_L','stop_frame_L','start_frame_R','stop_frame_R','label']
    dataset['item']=dataset.index
    #transform labels in integers
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(dataset['label'])
    dataset['label']=targets

    return dataset,le #le necessary to transform ints to labels

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

def get_values_spectr(dataset,idx,emg_data,emg_data2):# get spectrograms and build an image with 16 channels
    
    x=[]
    for index in idx:
        startL=int(dataset.iloc[int(index)]['start_frame_L'])
        stopL=int(dataset.iloc[int(index)]['stop_frame_L'])
        tmp= emg_data[startL:stopL]
        startR=int(dataset.iloc[int(index)]['start_frame_R'])
        stopR=int(dataset.iloc[int(index)]['stop_frame_R'])
        tmpR= emg_data2[startR:stopR]
        tmp_spec=visualize_spec.compute_spectrogram(emg_data[startL:stopL],"first_plot")
        tmp_spec2=torch.cat((tmp_spec,visualize_spec.compute_spectrogram(emg_data2[startR:stopR],"first_plot")),0)

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

def main():
    num_classes=20 #do not consider base class 0
    device = torch.device("mps" if torch.has_mps else "cpu")#conv3D is still not working on mps, leave cpu
    batch_size=16
    #learning_rate=0.001 #0.0001
    #epochs=20
    #momentum=0.9
    #weight_decay=0.000001 #0.000001
    emg_L, time_s_L,time_str_L= import_data(0)
    emg_R,time_s_R,time_str_R= import_data(1)
    stifness_L=transform_data(emg_L,time_s_L,False)
    stifness_R=transform_data(emg_R,time_s_R,False)
    dataset_info,le=create_dataset(time_s_L,time_s_R)
    #dataset_info,le=create_dataset(time_s_L,emg_L,time_str_L)
    
    ''''''
    X_train,X_test,y_train,y_test=train_test_split(dataset_info['item'],dataset_info['label'], test_size=0.3,random_state=42)
    loader_train= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_train)), torch.tensor(np.array(y_train))), shuffle=True, batch_size=batch_size,drop_last=True)
    loader_test= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_test)), torch.tensor(np.array(y_test))), shuffle=False, batch_size=batch_size,drop_last=True)



main()
