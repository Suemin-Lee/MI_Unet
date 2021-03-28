#!/usr/bin/env python


import tensorflow as tf
import os
import random
import numpy as np
from collections import OrderedDict

from tqdm import tqdm 
import keras
import keras.backend as K
import numpy as np

import utils
import loggingreporter 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt



import os
if not os.path.exists('plots/'):
    os.mkdir('plots')

from six.moves import cPickle
from collections import defaultdict, OrderedDict

import kde
import simplebinmi




seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.int32)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.int32)
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.int32)


# Which measure to plot
infoplane_measure = 'upper'
#infoplane_measure = 'bin'

DO_SAVE        = True    # Whether to save plots or just show them
DO_LOWER       = False    # (infoplane_measure == 'lower')   # Whether to compute lower bounds also
DO_BINNED      = True    #(infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning
FULL_MI        = True
DO_SAVE_GRAD   = True    


MAX_EPOCHS = 10000     # Max number of epoch for which to compute mutual information measure
COLORBAR_MAX_EPOCHS = MAX_EPOCHS

# Directories from which to load saved layer activity
ARCH ='128-64-32-16-8-16-32-64-10000'

DIR_TEMPLATE = '%%s_%s'%ARCH

# Functions to return upper and lower bounds on entropy of layer activity
noise_variance = 2e-1                    # Added Gaussian noise variance
Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

# nats to bits conversion factor
nats2bits = 1.0/np.log(2) 


PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted 




# Data structure used to store results
measures = OrderedDict()
measures['relu'] = {}
# measures['tanh'] = {}




#  Resizing image
print('Resizing training images and masks for training')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_train[n] = mask   

# test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images and masks for testing') 
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)  
            
    Y_test[n] = mask   

print('Resizing images... Done!')




# Normalization of the images
X_train= X_train/255
Y_train= Y_train/255
X_test = X_test/255
Y_test = Y_test/255




from collections import namedtuple

nb_classes=2**16;  #Total number of classifier types (2**16= 65,536) 
Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])

y_train = Y_train
y_test = Y_test

trn = Dataset(X_train, Y_train, y_train, nb_classes)
tst = Dataset(X_test , Y_test, y_test, nb_classes)

y = tst.y
Y = tst.Y

if FULL_MI:
    full = utils.construct_full_dataset(trn,tst)
    y = full.y
    Y = full.Y



# Spatial coarsening methods 

# create classifier types
mask = ((np.arange(2**16)[:,None] & (1 << np.arange(16))) != 0)
classifer = mask.astype(int).reshape(-1,4,4)
# coarsening output image into 4*4 pixel
import skimage.measure
import statistics

division = 16
thresholdT = 32*32/division #Threshold value

# down sampling
block_size = 32
coarsening = skimage.measure.block_reduce(full.y[1], block_size =(block_size,block_size,1), func=np.sum)

for i in range(full.y.shape[0]-1):
    down_sample = full.y[i+1]
    tmp = skimage.measure.block_reduce(down_sample, block_size =(block_size,block_size,1), func=np.sum)
    coarsening = np.concatenate((coarsening, tmp))

coarsening[coarsening <= thresholdT] = 0  # if the selected area is smaller than the thresholdT masks as zero
coarsening[coarsening > thresholdT] = 1   # if the selected area is larger than the thresholdT masks as one

coarsening = coarsening.reshape(full.y.shape[0],4,4)

# Make 2d into 1d label
y_1dim =[]
for i in range(full.y.shape[0]):
    tmp = coarsening[i] 
    for j in range(nb_classes):
        if np.all(tmp == classifer[j]):
            y_1dim.append(j)
y_1dim = np.array(y_1dim)

# label postion similar to MNIST data setting
saved_labelixs = {}
for i in range(nb_classes):
    saved_labelixs[i] = y_1dim == i

# calculate label probability P(Y=y) which needs for I(M;Y) (similar to MNIST data setting)
labelprobs=[]
for i in range(nb_classes):
    tmp = sum(saved_labelixs[i])/(full.y.shape[0])
    labelprobs.append(tmp)
labelprobs = np.array(labelprobs)




#Compute Mutual Information (similar to MNIST data setting with different number of classifiers)

for activation in measures.keys():

    cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
    if not os.path.exists(cur_dir):
        print("Directory %s not found" % cur_dir)
        continue
        
    # Load files saved during each epoch, and compute MI measures of the activity in that epoch
    print('*** Doing %s ***' % cur_dir)
    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            continue
        fname = cur_dir + "/" + epochfile
        with open(fname, 'rb') as f:
            d = cPickle.load(f)
        
            epoch = d['epoch']
        if epoch in measures[activation]: # Skip this epoch if its already been processed
            continue                      # this is a trick to allow us to rerun this cell multiple times)
            
        if epoch > MAX_EPOCHS:
            continue

        print("Doing", fname)
        
        num_layers = len(d['data']['activity_tst'])
        grad_mean = d['data']['gradmean']
        grad_std = d['data']['gradstd']
        
        if PLOT_LAYERS is None:
            PLOT_LAYERS = []
            for lndx in range(num_layers):
                #if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                PLOT_LAYERS.append(lndx)

        cepochdata = defaultdict(list)
        for lndx in range(num_layers):
            activity = d['data']['activity_tst'][lndx]
   
            if len(activity)>3: #if more than 2 dimensional change into 1 dimensional matrix (3D into 1D matrix)
                                
                activity=np.reshape(activity , [activity.shape[0] , -1])
            
            # Compute marginal entropies
            h_upper = entropy_func_upper([activity,])[0]
            if DO_LOWER:
                h_lower = entropy_func_lower([activity,])[0]
                
            # Layer activity given input. This is simply the entropy of the Gaussian noise
            hM_given_X = kde.kde_condentropy(activity, noise_variance)

            # Compute conditional entropies of layer activity given output
            hM_given_Y_upper=0.
            for i in range(nb_classes):
                data = activity[saved_labelixs[i],:]
                if len(data)==0:
                    hcond_upper =0
                else: hcond_upper = entropy_func_upper([data,])[0]
                hM_given_Y_upper += labelprobs[i] * hcond_upper

            
#             if DO_LOWER:
#                 hM_given_Y_lower=0.
#                 hM_given_Y_lower_ls=[]
#                 for i in range(nb_classes):
#                     hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
#                     hM_given_Y_lower += labelprobs[i] * hcond_lower
        
                
            cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
            cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
            cepochdata['H_M_upper'  ].append( nats2bits * h_upper )

            cepochdata['H_M_given_X_upper'].append( nats2bits * (hM_given_X) )
            cepochdata['H_M_given_Y_upper'].append( nats2bits * (hM_given_Y_upper) )

            pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
            pstr += ' | Conditional : H(M|X)=%0.3f, H(M|Y)=%0.3f' % (cepochdata['H_M_given_X_upper'][-1], cepochdata['H_M_given_Y_upper'][-1])
# 
#             if DO_LOWER:  # Compute lower bounds
#                 cepochdata['MI_XM_lower'].append( nats2bits * (h_lower - hM_given_X) )
#                 cepochdata['MI_YM_lower'].append( nats2bits * (h_lower - hM_given_Y_lower) )
#                 cepochdata['H_M_lower'  ].append( nats2bits * h_lower )
#                 pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

            if DO_BINNED: # Compute binning estimates
                binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.1)
                cepochdata['MI_XM_bin'].append( nats2bits * binxm )
                cepochdata['MI_YM_bin'].append( nats2bits * binym )
                pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])
            
            print('- Layer %d %s' % (lndx, pstr) )

            
        measures[activation][epoch] = cepochdata
        




# MI versus epochs Plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style('darkgrid')

plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(4,2)
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    if not len(epochs):
        continue

    
    plt.subplot(gs[0,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        xmvalsU = np.array([vals[epoch]['MI_XM_upper'][layerid] for epoch in epochs])
        plt.plot(epochs, xmvalsU, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.ylabel('I(X;M)')

    
    plt.subplot(gs[1,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        YmvalsU = np.array([vals[epoch]['MI_YM_upper'][layerid] for epoch in epochs])
        plt.plot(epochs, YmvalsU, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.ylabel('I(Y;M)')
    
    plt.subplot(gs[2,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
        plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
    plt.xlabel('Epoch')
    plt.ylabel("I(X;M)bin")

    plt.subplot(gs[3,actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        hbinnedvals = np.array([vals[epoch]['MI_YM_bin'][layerid] for epoch in epochs])
        plt.semilogx(epochs, hbinnedvals, label='Layer %d'%layerid)
    plt.xlabel('Epoch')
    plt.ylabel("I(Y;M)bin")
    
if DO_SAVE:
    plt.savefig('plots/' + DIR_TEMPLATE % ('Epochs_over_MI_on_different_layers')+'.pdf',bbox_inches='tight')




#------ Modified infoplane 23 layers of 4*6 subplot  ------------  
max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []


fig=plt.figure(figsize=(12,8))
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    if not len(epochs):
        continue
    for lndx, layerid in enumerate(PLOT_LAYERS):
        xmvals=[]; ymvals=[]; c_save=[]
        
        if len(PLOT_LAYERS)>6:
            if lndx<5:
                plt.subplot(4,int(len(PLOT_LAYERS)/4)+1,lndx+1)   
            elif lndx>3:
                plt.subplot(4,int(len(PLOT_LAYERS)/4)+1,lndx+1)
            
            
            for epoch in epochs:
                c = sm.to_rgba(epoch)
                xmval = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[layerid]
                ymval = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[layerid]
                xmvals.append(xmval)
                ymvals.append(ymval)
                c_save.append(c)
                max_y = np.amax(ymval)
                max_x = np.amax(xmval)
                
            for i,c in enumerate(c_save):
                plt.plot(xmvals[i], ymvals[i], c=c, alpha=0.1, zorder=1)
                plt.scatter(xmvals[i], ymvals[i], s=20, edgecolor='none', zorder=2,c=c)
                plt.ylim([0, max_y+1])
                plt.xlim([0, max_x+1])
                plt.title(' Layer %1.0f'%(layerid+1),size=20)
                if lndx%6==0:
                    plt.ylabel('I(Y;M)',size=20)
                if lndx>16:
                    plt.xlabel('I(X;M)',size=20)

                
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
cb = plt.colorbar(sm, cax=cbaxes)
cb.ax.tick_params(labelsize='large')
cb.set_label(label='Epoch',size=20)
plt.tight_layout()


if DO_SAVE:
    plt.savefig('plots/' + DIR_TEMPLATE % ('Infoplane_layers_subplot')+'.pdf',bbox_inches='tight')
    
    



# U-shaped plot
from matplotlib.colors import LogNorm

max_epoch = max( (max(vals.keys()) if len(vals) else 0) for vals in measures.values())
# sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm = plt.cm.ScalarMappable(cmap='gnuplot',norm=LogNorm(vmin=1, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []


fig=plt.figure(figsize=(8,4))
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    c_save=[]
    if len(epochs) <10: new_epochs = epochs
    else : new_epochs= np.array([1,10,100,500,1000,2000,5000,9900])

    for epoch in new_epochs:
        c = sm.to_rgba(epoch)
        c_save.append(c)
        MI_XM = np.array([vals[epoch]['MI_XM_upper'][layerid] for lndx, layerid in enumerate(PLOT_LAYERS)])
        MI_YM = np.array([vals[epoch]['MI_YM_upper'][layerid] for lndx, layerid in enumerate(PLOT_LAYERS)])
        
        plt.subplot(1,2,1)    
        for i,c in enumerate(c_save):
            plt.plot(PLOT_LAYERS, MI_XM, c=c, alpha=0.1, zorder=1)
            plt.scatter(PLOT_LAYERS, MI_XM, s=20, edgecolor='none', zorder=2,c=c)
            plt.ylabel('I (X;M)',size=18)
            plt.xlabel('Layers',size=18)
            plt.title('MI_XM_upper')

        plt.subplot(1,2,2)    
        for i,c in enumerate(c_save):
            plt.plot(PLOT_LAYERS, MI_YM, c=c, alpha=0.1, zorder=1)
            plt.scatter(PLOT_LAYERS, MI_YM, s=20, edgecolor='none', zorder=2,c=c)
            plt.ylabel('I (Y;M)',size=18)
            plt.xlabel('Layers',size=18)
            plt.title('MI_YM_upper')
            
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
cb = plt.colorbar(sm, cax=cbaxes)
cb.ax.tick_params(labelsize='large')
cb.set_label(label='Epoch',size=18)

plt.tight_layout()

if DO_SAVE:
    plt.savefig('plots/' + DIR_TEMPLATE % ('U_shaped_MI_')+'.pdf',bbox_inches='tight')





