import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet34
import torchvision
import numpy as np
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
import pickle
# some_file.py
import sys
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './actionNet')
filepath=''
annotations_ActionNet=pd.DataFrame()
import visualize_spec


#from actionNet import visualize_spec
#import librosa 
class AlexNet(torch.nn.Module): #Module is base class for all neural network modules.
  def __init__(self,n_classes=20) :
    super(AlexNet,self).__init__()
    self.layer1=nn.Sequential(
                nn.Conv2d(3,96,kernel_size=11,stride=4,padding=0),#1st question, which filter I gave?? 1 0 -1 x nrows always?
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
        nn.Linear(9216,4096),
        nn.ReLU())
    self.fc2= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096,4096),
        nn.ReLU())
    self.fc3= nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(4096,n_classes),
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
class ResNet16(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,x):
        output=""
        return output
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

def train(model,loader,dataset,emg_dataL,emg_dataR,optimizer,loss_function,device):
    model.train()
    samples=0.
    cumulative_loss=0.
    cumulative_accuracy=0.
    cumulative_accuracytop5=0.
    for batch_idx,(idx,y) in enumerate(loader):
        #trasformare ixd in X
        #_,X=get_values(dataset,idx,emg_time_s,emg_data)
        #X=get_values(dataset,idx,emg_dataL) #working with time --> 
        
        #X=torch.Tensor(emg_data[int(dataset.iloc[idx]['start_frame']):int(dataset.iloc[idx]['stop_frame'])])
        X=get_values_spectr(dataset,idx,emg_dataL,emg_dataR) #has to return the spectrgorams
        X=X.reshape(-1,X.size(3),X.size(2))# X.size(3)->seq_len; X.size(2)->inpout_len
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
    return cumulative_loss/samples, cumulative_accuracy/samples*100, cumulative_accuracytop5/samples*100

def test(model,loader,dataset,emg_dataL,emg_dataR,loss_function,device):
    model.eval()
    samples=0.
    cumulative_loss=0.
    cumulative_accuracy=0.
    cumulative_accuracytop5=0.
    with torch.no_grad():
        for batch_idx,(idx,y) in enumerate(loader):
            #trasformare ixd in X
            #_,X=get_values(dataset,idx,emg_time_s,emg_data)
            #X=get_values(dataset,idx,emg_dataL) #working with time --> 
            
            #X=torch.Tensor(emg_data[int(dataset.iloc[idx]['start_frame']):int(dataset.iloc[idx]['stop_frame'])])
            X=get_values_spectr(dataset,idx,emg_dataL,emg_dataR) #has to return the spectrgorams
            X=X.reshape(-1,X.size(3),X.size(2))# X.size(3)->seq_len; X.size(2)->inpout_len
            X,y=X.to(device),y.to(device)
            outputs=model(X)
            loss=loss_function(outputs,y)
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            samples += idx.shape[0]
            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            #_, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(y).sum().item()
            cumulative_accuracytop5 += get_accuracytop5(outputs,y)
    return cumulative_loss/samples, cumulative_accuracy/samples*100, cumulative_accuracytop5/samples*100
    
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
    global filepath,annotations_ActionNet #= '../ActionNet_data/2022-06-14_16-38-43_streamLog_actionNet-wearables_S04.hdf5'
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
        tmp=[(start,end) for start,end in zip(label_start_times_s,label_end_times_s)] #working for 1 sample per action
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
        #cresting annotation_for ActionNet
        df_ann=pd.DataFrame()
        df_ann=pd.concat([df_ann,pd.DataFrame(tmp)])
        df_ann=pd.concat([df_ann,pd.DataFrame([target_label]*len(label_start_times_s))],axis=1)
        annotations_ActionNet=pd.concat([annotations_ActionNet,df_ann])
        
    dataset.reset_index(drop=True,inplace=True)
    dataset.columns=['start_frame_L','stop_frame_L','start_frame_R','stop_frame_R','label']
    dataset['item']=dataset.index

    annotations_ActionNet.reset_index(drop=True,inplace=True)
    annotations_ActionNet.columns=['start_time','stop_time','label']
    annotations_ActionNet['item']=annotations_ActionNet.index
    #transform labels in integers
    le = preprocessing.LabelEncoder()
    targets = le.fit_transform(dataset['label'])
    dataset['label']=targets

    le_ann=preprocessing.LabelEncoder()
    targets_ann = le_ann.fit_transform(annotations_ActionNet['label'])
    annotations_ActionNet['class']=targets_ann
    annotations_ActionNet.to_pickle("annotations_ActionNet.pkl")
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
def main():
    sequence_len=26 #params for output images of spectrogram
    input_len=17
    hidden_size=5
    num_layers=2
    num_classes=20 #do not consider base class 0
    device = torch.device("mps" if torch.has_mps else "cpu")#conv3D is still not working on mps, leave cpu
    batch_size=8
    learning_rate=0.001
    epochs=50
    momentum=0.5
    weight_decay=0.000001
    emg_L, time_s_L,time_str_L= import_data(0)
    emg_R,time_s_R,time_str_R= import_data(1)
    stifness_L=transform_data(emg_L,time_s_L,False)
    stifness_R=transform_data(emg_R,time_s_R,False)
    dataset_info,le=create_dataset(time_s_L,time_s_R)
    #dataset_info,le=create_dataset(time_s_L,emg_L,time_str_L)
    X_train,X_test,y_train,y_test=train_test_split(dataset_info['item'],dataset_info['label'], test_size=0.2,random_state=1234)
    loader_train= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_train)), torch.tensor(np.array(y_train))), shuffle=True, batch_size=batch_size,drop_last=True)
    loader_test= data.DataLoader(data.TensorDataset(torch.tensor(np.array(X_test)), torch.tensor(np.array(y_test))), shuffle=False, batch_size=batch_size,drop_last=True)

    loss_function=get_loss_function()
    model=EMG_Classifier(input_len,hidden_size,num_layers,num_classes,batch_size) #check correttezza implementazione rete
    model_Resnet18= resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1) #use pretrained 2dCNN
    model.to(device)
    optimizer=get_optimizer(model,learning_rate,weight_decay,momentum,1)
    print(model)
    val_loss_list = []
    val_accuracy_list = []
    train_loss_list = []
    train_accuracy_list = []
    for epoch in range(epochs):
        
        train_loss,train_accuracy, train_acc_top5 =train(model,loader_train,dataset_info,stifness_L,stifness_R,optimizer,loss_function,device)#inserire il data loader nel processo, seguire esempi per il training
        test_loss,test_accuracy, test_acc_top5=test(model,loader_test,dataset_info,stifness_L,stifness_R,loss_function,device)#
        val_loss_list.append(test_loss)
        val_accuracy_list.append(test_accuracy)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        
        #test()#ricordarsi di usare torch.no_grad e poi procedere al test e incrociare le dita :D
        
        print("-"*40)
        print('Epoch: {:d}'.format(epoch+1))
        print('\t Training loss {:.5f}, Training accuracy top1 {:.2f},Training accuracy top5 {:.2f}'.format(train_loss,train_accuracy,train_acc_top5))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
        # val_accuracy))
        print("-"*40)
        # print("After training:")
        # train_loss, train_accuracy = test(model, train_loader, loss_function,device)
        # val_loss, val_accuracy = test(model, val_loader, loss_function,device)
        # test_loss, test_accuracy = test(net, test_loader, loss_function,device)
        # print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
        # train_accuracy))
        # print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
        # val_accuracy))
        print('\t Test loss {:.5f}, Test accuracy top1 {:.2f}, Test accuracy top5 {:.2f}'.format(test_loss, test_accuracy,test_acc_top5))
        print('-'*40)
    save_path = 'custom-classifier_LSTM_final_10_last_tr_epochs.pth'
    torch.save(model.state_dict(), save_path)
    plot_accuracy(train_accuracy_list,val_accuracy_list,"LSTM-Spectrogram 16channels")
    return
def plot_accuracy(train_accuracy,val_accuracy,model_name=""):
    import seaborn as sns
    sns.set_theme()
    palette = sns.color_palette('pastel')       
    fig,ax = plt.subplots(figsize=(10,10))
    ax.set_title(f"Train-Validation Accuracy - model: {model_name}")
    sns.lineplot(x=list(range(1,len(train_accuracy)+1)), y=train_accuracy,marker='o', label='train',ax=ax)# , col="Frames",height=4, aspect=0.6,)
    sns.lineplot(x=list(range(1,len(train_accuracy)+1)), y=val_accuracy,marker='o', label='test',ax=ax)# , col="Frames",height=4, aspect=0.6,)
    ax.set_xlabel('num_epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    fig.savefig("accuracy_LSTM_50 epochs.pdf")
def get_loss_function():
    loss_function= torch.nn.CrossEntropyLoss()
    return loss_function

def get_optimizer(net,lr,wd,momentum,kind=0):
    if kind ==0:
        optimizer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd,momentum=momentum)
    else:
        optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    return optimizer

main()
