import pickle
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
import torchvision.transforms as T
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kmeans_pytorch import kmeans # import kmeans-pytorch using pip install kmeans-pytorch on env; then install tqdm 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.pipeline import make_pipeline
import seaborn as sns
from PIL import Image
import pandas as pd

modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)
X=[]
X_torch=[]
true_labels=[]
nouns=[]
saved_feat_path=""
images_path="../ek_data/frames"
annotations=[]
def init():
    global X,X_torch,true_labels,nouns,saved_feat_path,annotations
    # saved_feat_path="./saved_features/change_me_D3_train.pkl"
    #saved_feat_path="./saved_features/D3_pkls_old/change_me_D3_test.pkl"
    saved_feat_path="./saved_features/featSaved_trainSplit_D1_train.pkl"
    # annotations_path="./train_val/D3_train.pkl" 
    annotations_path="./train_val/D1_test.pkl" 

    file=open(saved_feat_path,'rb')
    feat=pickle.load(file)
    annotations=pickle.load(open(annotations_path,'rb'))
    true_labels=annotations['verb_class']
    nouns=[n.split(' ')[-1] for n in annotations['narration']]
    x=[]
    i=0
    uids=[]
    for record in feat['features']:
        x.append(record['features_RGB'][2])
        uids.append(record['uid'])
    X=np.array(x)
    X_torch=torch.from_numpy(X)
    print(f"size of X features: {X.shape}")
def main():
    # global variables among training functions
   init()
   performKmeans(X,X_torch,8,true_labels)

def performKmeans(X,X_torch,num_clusters,true_labels):

    # kmeans with pytorch
    global saved_feat_path
    start=time.time()
    #num_clusters=8
    cluster_ids_x, cluster_centers = kmeans(
        X=X_torch, num_clusters=num_clusters, distance='euclidean', device=torch.device('mps')
    )
    end=time.time()
    print(f"time required for pytorch kmeans: {end-start}")

    # kmeans with scikitlearn
    start=time.time()
    kmeans_SL_0 = KMeans(n_clusters=num_clusters, random_state=0,init="k-means++", n_init='auto').fit(X)
    end=time.time()
    print(f"time required for scikit_learn kmeans: {end-start}")
    
    # silhouette calculation
    silh=silhouette_score(X,kmeans_SL_0.labels_)
    silh_torch=silhouette_score(X,cluster_ids_x)
    print(f"silhouette for kmeans_sklearn: {silh}")
    print(f"silhouette for kmeans_torch: {silh_torch}")

    ############ PCA & PLOT POINTS #################
    random_state=0
    # with PCA
    model=make_pipeline(StandardScaler(),PCA(2,random_state=random_state))
    X_2feat=model.fit_transform(X)
    
    ################################################
    

    #########  T-SNE & PLOT #######################
    model_tSNE=make_pipeline(StandardScaler(),TSNE())
    tsne=TSNE()
    #X_tsne=tsne.fit_transform(X)
    X_tsne=model_tSNE.fit_transform(X)
    ###############################################


    kmeans_SL = KMeans(n_clusters=num_clusters, random_state=0,init="k-means++", n_init='auto').fit(X_2feat)
    kmeans_tSNE = KMeans(n_clusters=num_clusters, random_state=0,init="k-means++", n_init='auto').fit(X_tsne)

    s=silhouette_score(X_2feat,kmeans_SL.labels_)
    s_tsne=silhouette_score(X_tsne,kmeans_tSNE.labels_)
    print(f"silhouette for kmeans_sklearn with PCA=2: {s} ")
    print(f"silhouette for kmeans_sklearn with tsne: {s_tsne} ")

    figname=(saved_feat_path.split('/')[-1]).split('.')[0]

    plot_cluster(X_2feat,X_tsne,kmeans_SL,kmeans_tSNE,true_labels,s,s_tsne)
    #plot_cluster_1image(X_2feat[:,0],X_2feat[:,1],kmeans_SL,s,figname)
    fig,ax=plt.subplots()
    #plot_scatterIm(X_2feat[:,0],X_2feat[:,1],8,kmeans_SL,ax,.16)
    plot_scatterIm(X_tsne[:,0],X_tsne[:,1],8,kmeans_tSNE,ax,.16)

    print("")
def plot_cluster(x,y,kmeans_pca,kmeans_tsne,true_labels,s_pca,s_tsne):

    fig, ax = plt.subplots(2)
    sns.scatterplot(x=x[:,0],y=x[:,1],c=kmeans_pca.labels_,ax=ax[0])
    # plt.scatter(x=X_2feat[:,0],y=X_2feat[:,1])
    ax[0].set_title(f"PCA 2 components - silh: {s_pca:.2f}")
    ax[0].plot()
    ax[0].scatter(
    kmeans_pca.cluster_centers_[:, 0],
    kmeans_pca.cluster_centers_[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    zorder=10,
    color='r'
    )
    # sns.scatterplot(x=X_2feat[:,0],y=X_2feat[:,1],c=true_labels,ax=ax[1])
    sns.scatterplot(x=y[:,0],y=y[:,1],c=kmeans_tsne.labels_,ax=ax[1])#for noun there's some work to do

    # plt.scatter(x=X_2feat[:,0],y=X_2feat[:,1])
    ax[1].set_title(f"t-SNE 2 components - silh: {s_tsne:.2f}")
    ax[1].plot()
    ax[1].plot()
    ax[1].scatter(
    kmeans_tsne.cluster_centers_[:, 0],
    kmeans_tsne.cluster_centers_[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    zorder=10,
    color='r'
    )
    print()
    return
def plot_cluster_1image(x,y,kmeans_SL,s,plot_name):
    global saved_feat_path
    fig, ax = plt.subplots(2)
    sns.scatterplot(x=x,y=y,c=kmeans_SL.labels_,ax=ax[0])
    # plt.scatter(x=X_2feat[:,0],y=X_2feat[:,1])
    ax[0].set_xlabel("PCA 0")
    ax[0].set_ylabel("PCA 1")
    plt.suptitle("Features distributions PCA 2 components")
    ax[0].set_title(f"8 clusters - silh: {s}")
    ax[0].plot()
    ax[0].scatter(
    kmeans_SL.cluster_centers_[:, 0],
    kmeans_SL.cluster_centers_[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    zorder=10,
    color='r'
    )
    figpath="./Images/"+plot_name+".pdf"
    fig.savefig(figpath)
    label='plate'
    plot_centerImage(label,x,y,ax[1])
    print()
    return
def plot_centerImage(label,x,y, ax):
    mask=pd.Series(nouns)==label
    mask=pd.Series(mask[mask].index)
    record=int(mask.sample(n=1,random_state=1234))
    c_frame=int((annotations.iloc[record]['start_frame']+annotations.iloc[record]['stop_frame'])/2)
    tmpl="img_{:010d}.jpg"
    img_path=os.path.join(images_path,
                          annotations.iloc[record]['video_id'],
                          tmpl.format(c_frame))
    img = Image.open(img_path).convert('RGB')
    transform=T.CenterCrop(224)#size of cropped starting images
    img=transform(img)
    ax.imshow(img)#
    print("how to print")

    return
def plot_scatterIm(x,y,labels,kmeans,ax,zoom=0.1):
    #get the images here#
    #sampling 1 image (and 1 sample for verb class)
    envs=['sink','microwave','fridge','dishwasher']
    mask=[]
    N=3 # number of images to plot
    for i in envs:#range(labels):
        #v_class=pd.Series(annotations['verb_class'])==i
        v_class=pd.Series(nouns)==i
        mask.append(pd.Series(v_class[v_class].index))
    #records=[int(m.sample(n=3,random_state=None)) if len(m)>2 else int(m.sample(n=len(m),random_state=None)) for m in mask]
    records=[m.sample(n=N,random_state=1234) if len(m)>2 else m.sample(n=len(m),random_state=None) for m in mask]
    
    recs=[]
    for rec in records:
        tmp=[int(r) for r in rec]
        recs.append(tmp)

    colors={envs[0]:'blue',envs[1]:'orange',envs[2]:'green',envs[3]:'red'}#,envs[4]:'red'}
    #colors={0:'blue',1:'orange',2:'green',3:'green',4:'red',5:'purple',6:'brown',7:'pink'}

    ####################
    image_paths=[]
    all_recs=[]
    for rec in recs:#records:
        for r in rec:
            c_frame=int((annotations.iloc[r]['start_frame']+annotations.iloc[r]['stop_frame'])/2)
            tmpl="img_{:010d}.jpg"
            img_path=os.path.join(images_path,
                            annotations.iloc[r]['video_id'],
                            tmpl.format(c_frame))
            image_paths.append(img_path)
            all_recs.append(r)
    x_=x[all_recs]
    y_=y[all_recs]
    lab=kmeans.labels_[all_recs]
    #cols=[colors[value] for value in lab]
    if ax is None:
        ax=plt.gca()

    # for xi,yi,r in zip(x,y,all_recs):
    #     ax.scatter(x=xi,y=yi,c=colors[nouns[r]])
    for r in recs:
        ax.scatter(x=x[r],y=y[r],c=colors[nouns[r[0]]])
    ax.legend(envs)
    ax.grid(True)
    # scatter=ax.scatter(x, y,c=colors.values())
    # legend1 = ax.legend(*scatter.legend_elements(num=6),loc="lower left", title="Environments")
    # ax.add_artist(legend1)
    #ax.legend(envs)
    #colors={0:'blue',1:'orange',2:'green',3:'green',4:'red',5:'purple',6:'brown',7:'pink'}
    for x0, y0, path,frame in zip(x_, y_,image_paths,all_recs):
        ab = AnnotationBbox(getImage(path,zoom=zoom), (x0, y0),bboxprops =dict(edgecolor=colors[nouns[frame]]), frameon=True)
        ax.add_artist(ab)
    ax.legend(envs)
def getImage(path, zoom=0.1):
    return OffsetImage(plt.imread(path), zoom=zoom)
#try with average clip
#try with DB Scan & GMM mixtures
def performDB_Scan():
    return

if __name__ == '__main__':
    main()
#