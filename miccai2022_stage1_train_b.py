### regular import functions

import os
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import zipfile
import shutil
import glob
import time
import copy
from PIL import Image

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import ImageGrid
from itertools import chain

#%%
### deep learning import functions

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import Tensor
#from torchsummary import summary
import torch.utils.data as data

import torchvision
from torchvision import models
from torchvision import transforms, datasets, models
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import easydict
from tqdm import tqdm # for python file
# from tqdm.notebook import tqdm
# from tqdm.notebook import trange

from video_dataset import VideoFrameDataset, ImglistToTensor
from video_dataset import plot_video, denormalize

#from audio_dataset import AudioDataset, spec_to_image
from auto_encoder_custom_v1 import Seq2Seq

import pandas as pd
import numpy as np


from torchvision.models import resnet18, ResNet18_Weights

#from torchsummary import summary

"""# seed and cuda setup"""
torch.cuda.empty_cache()

# random seed assign
seed = 2021
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device found and selected: {device}')



def main():

    modality = 'body' # face, body
    fold = '0'

    # if train = True:
        # model training
    # if train =False:
        # generate latent features for training, validation, testing dataset
    train= True

    # ## server path
    videos_root      = "/data/jiayiwang/Neonate_Pain_Assessment/Testing_BiliearCNNLSTM/folds/"
    videos_root_post = "/data/jiayiwang/Neonate_Pain_Assessment/Testing_BiliearCNNLSTM"



    annotation_file_vd_train = "{0}{1}/reduceV/fold{2}_bodytrain.txt".format(videos_root, modality, fold)
    annotation_file_vd_valid = "{0}{1}/reduceV/fold{2}_bodytrain.txt".format(videos_root, modality, fold)
    annotation_file_vd_test  = "{0}{1}/reduceV/fold{2}_bodytest.txt".format(videos_root, modality, fold)

    print(annotation_file_vd_train)
    print(annotation_file_vd_test)

    # data are already 224*224. need to add for baseline 64*64
    image_size_vd_frame = 224 #64
    image_size_vd = (image_size_vd_frame, image_size_vd_frame)

    # frame extraction process
    num_segments = 30
    frames_per_segment = 1
    seq_length = num_segments * frames_per_segment


    # data preprocess - train
    preprocess_vd_train = transforms.Compose([
        # need to convert tensor first
        ImglistToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imageNet
    ])


    # data preprocess - test/valid
    preprocess_vd_test = transforms.Compose([
        ImglistToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imageNet
    ])



    #################
    # training set
    #################


    # dataset read
    dataset_vd_train = VideoFrameDataset(
        root_path = videos_root_post,
        annotationfile_path = annotation_file_vd_train,
        num_segments = num_segments,
        frames_per_segment = frames_per_segment,
        imagefile_template = 'videoframe_{:05d}.jpg',
        transform = preprocess_vd_train,
        random_shift = train,
        test_mode = -train
    )

    #################
    # validation set
    #################

    # dataset read
    dataset_vd_valid = VideoFrameDataset(
        root_path = videos_root_post,
        annotationfile_path = annotation_file_vd_valid,
        num_segments = num_segments,
        frames_per_segment = frames_per_segment,
        imagefile_template = 'videoframe_{:05d}.jpg',
        transform = preprocess_vd_test, # same to test processing
        random_shift = False,
        test_mode = True
    )

    #################
    # testing set
    #################

    # dataset read
    dataset_vd_test = VideoFrameDataset(
        root_path = videos_root_post,
        annotationfile_path = annotation_file_vd_test,
        num_segments = num_segments,
        frames_per_segment = frames_per_segment,
        imagefile_template = 'videoframe_{:05d}.jpg',
        transform = preprocess_vd_test,
        random_shift = False,
        test_mode = True
    )



    #%%
    ### batch data loading
    if train:
        batch_size       = 16
        batch_size_valid = 8
        batch_size_test  = 1
    else:
        batch_size       = 1
        batch_size_valid = 1
        batch_size_test  = 1



    # training - video
    dataloader_vd_train = torch.utils.data.DataLoader(dataset=dataset_vd_train, batch_size=batch_size, shuffle=True,
                                                      num_workers=0, pin_memory=True, drop_last=False)

    dataloader_vd_valid = torch.utils.data.DataLoader(dataset=dataset_vd_valid, batch_size=batch_size_valid, shuffle=True,
                                                      num_workers=0, pin_memory=True, drop_last=False)

    dataloader_vd_test  = torch.utils.data.DataLoader(dataset=dataset_vd_test, batch_size=batch_size_test, shuffle=False,
                                                     num_workers=0, pin_memory=True, drop_last=False)
    ## import model
    weights = ResNet18_Weights.DEFAULT
    rn18 = models.resnet18(weights = weights)

    ## check layers
    children_counter = 0
    for n,c in rn18.named_children():
        print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # Load the pretrained model
    weights = ResNet18_Weights.DEFAULT
    model_df = models.resnet18(weights = weights)

    # Use the model object to select the desired layer
    layer = model_df._modules.get('avgpool')

    # Image transforms if required - no need here already done in the videoloader
    scaler = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    args = easydict.EasyDict({
        "batch_size": batch_size, ## batch size setting
        "device": torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
        "input_size": 512, ## input dimension setting (resnet18 - 512)
        "hidden_size": 128, ## Hidden dimension setting
        "output_size": 512, ## output dimension setting
        "num_layers": 2,     ## number of LSTM layer
        "learning_rate" : 0.01, ## learning rate setting
        "max_iter" :10, ## max iteration setting
    })


    ## Load model
    model = Seq2Seq(args)
    # model= nn.DataParallel(model)
    model.to(args.device)

    # optimizer Setting
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, verbose = True)
    if train:
        model = run(args, model, dataloader_vd_train, dataloader_vd_valid, model_df, optimizer, layer, fold, modality,scheduler)
    else:
        model_path = videos_root_post +'/saved/reduceV/best-model/'+ fold + '-' + modality +'-resnet18-best-model-epoch-000-loss-0.32.pth'
        test(model_path, model, optimizer, args, dataloader_vd_test, layer,model_df,modality,fold,type)

def get_vector(image_name, layer, model_df): # image input
    img = image_name  # direct value load

    t_img = Variable(img).unsqueeze(0) # without transformation

    #t_img tensor is converted to a PyTorch 'Variable' object and unsqueezed along toe 0th dimension using 'unsqueeze(0)'

    my_embedding = torch.zeros(512) # The 'avgpool' layer has an output size of 512

    def copy_data(m, i, o):
      my_embedding.copy_(o.data.reshape(o.data.size(1)))

    h = layer.register_forward_hook(copy_data)
    model_df(t_img)
    h.remove()
    return my_embedding


## resnet imagenet for body
def get_vector_vd(input_batch_video, layer, model_df): # batch input
    feature_batch = []
    video_count = 0
    for video in input_batch_video:
        video_count += 1
        feature_video = []
        img_count = 0
        for image in video:
            img_count += 1
            feature_image = get_vector(image, layer, model_df)
            feature_video.append(torch.squeeze(feature_image))
        #print("total img is {}".format(img_count))
        feature_video = torch.stack(feature_video)
        feature_batch.append(feature_video)
    feature_batch = torch.stack(feature_batch)
    #print("Total video count is {}".format(video_count))
    return feature_batch


### training-testing utility function

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

## created for loading model
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min

def run(args, model, train_loader, test_loader, model_df, optimizer, layer, fold, modality,scheduler):

    # checkpoint initialization
    valid_loss_min = np.Inf

    epochs = range(args.max_iter) # should be range

    ## Training
    count = 0
    epoch_train_loss = []
    epoch_valid_loss = []
    learning_rate = []
    for epoch in epochs:
        print('epoch = {0} at time {1}'.format(epoch, time.ctime()))
        # for param in model.parameters():
        #     print("param.data is ")
        #     print(param.data)
        print('----------------------------------------------------------------------------------------------------------------')

        ## train
        print('training starts.....................')
        model.train()
        #set the model in training mode
        train_loss = 0.0
        optimizer.zero_grad()
        train_iterator = tqdm(enumerate(train_loader), total=len(train_loader), desc="train")

        for i, batch_data in train_iterator:
            print("Batch data {0}".format(len(batch_data)))
            # batch_data: 1. path, 2. tensor, 3. label
            _, dataV, label = batch_data
            print('enter batch loader')
            print(dataV.shape)
            ## deep features
            past_data = get_vector_vd(dataV, layer, model_df)
            past_data = past_data.to(args.device)
            label = label.to(args.device)
            print("past_data.shape is {}".format(past_data.shape))
            model(past_data,label)
            reconstruct_loss,prediction = model(past_data,label)
            ## Composite Loss
            loss = reconstruct_loss
            train_loss += loss.mean().item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_iterator.set_postfix({
                "train_loss": float(loss),
            })
        train_loss = train_loss / len(train_loader)
        epoch_train_loss.append(train_loss)
        scheduler.step(train_loss)
        learning_rate.append(optimizer.param_groups[0]["lr"])
        print("Train Score : [{}]".format(train_loss))
        print("Traing learning rate is {}".format(optimizer.param_groups[0]["lr"]))
        ## test
        print('validation starts.....................')
        model.eval()
        eval_loss = 0.0
        test_iterator = tqdm(enumerate(test_loader), total=len(test_loader), desc="test")
        with torch.no_grad():
            for i, batch_data in test_iterator:
                _, dataV, label = batch_data # Intensity-CBFEV-PainLabel

                print('enter batch loader')
                print(dataV.shape)

                ## deep features
                past_data = get_vector_vd(dataV, layer, model_df)

                past_data = past_data.to(args.device)
                label = label.to(args.device)
                reconstruct_loss,prediction = model(past_data,label)

                ## Composite Loss
                loss = reconstruct_loss

                eval_loss += loss.mean().item()

                test_iterator.set_postfix({
                    "eval_loss": float(loss),
                })

        eval_loss = eval_loss / len(test_loader)
        epoch_valid_loss.append(eval_loss)
        print("Evaluation Score : [{}]".format(eval_loss))

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, eval_loss))

        # create checkpoint variable and add important data. helps to resume the training after pause
        checkpoint = {
            'epoch': epoch,
            'valid_loss_min': eval_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }


        ## checkpoint dynamic saving filename
        parent_save_path ="/data/jiayiwang/Neonate_Pain_Assessment/Testing_BiliearCNNLSTM/saved/reduceV/v1"
        experiment_type = '-resnet18'
        checkpoint_path = parent_save_path + '/checkpoint/'+ fold + '-'+ modality + experiment_type +'-last-checkpoint.pth'
        best_model_path = parent_save_path + '/best-model/'+ fold + '-'+ modality + experiment_type +'-best-model' \
                          '-epoch-' + str(epoch).zfill(3) + '-loss-'+ str("{:.2f}".format(eval_loss)) \
                          +'.pth'
        print('Current time is {0}'.format(time.ctime()))
        ## save checkpoint
        #save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        if eval_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, eval_loss))

            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = eval_loss
    #save train_loss and valid_loss into csv file
    main_path = "/data/jiayiwang/Neonate_Pain_Assessment/Testing_BiliearCNNLSTM/saved/reduceV/v1/"
    pic_save = main_path + '{0}_loss.png'.format(modality)
    final_loss = pd.DataFrame({'train_loss':epoch_train_loss, 'valid_loss':epoch_valid_loss})
    loss_save_path = main_path + '/' + modality + '-' + fold + '-' + 'final_loss.csv'
    lr = pd.DataFrame({'learningRate':learning_rate})
    lr_save_path = main_path + '/' + modality + '-' + fold + '-' + 'learningrate.csv'
    final_loss.to_csv(loss_save_path)
    lr.to_csv(lr_save_path)
    #plot train_loss and valid_loss
    plt.clf()
    plt.plot(epoch_train_loss, color = 'blue',label = 'train_loss')
    plt.plot(epoch_valid_loss, color = 'red',label = 'valid_loss')
    plt.ylabel(' Loss')
    plt.xlabel('Epochs')
    plt.xticks(range(0,9),range(1,10))
    plt.text(0.8, 1.1, 'Train_Loss', color='blue', transform=plt.gca().transAxes)
    plt.text(0.8, 1.05, 'Valid_Loss', color='red', transform=plt.gca().transAxes)
    plt.savefig(main_path + '/'+'{0}_loss.png'.format(modality))

    return model



def test(model_path, model, optimizer, args, dataloader_vd_test, layer,model_df,modality,fold,type):
    model, optimizer, last_epoch, valid_loss_min = load_ckp(model_path, model, optimizer)
    print("optimizer = ", optimizer)
    print("last_epoch = ", last_epoch)
    print("valid_loss_min = ", valid_loss_min)
    test_iterator = tqdm(enumerate(dataloader_vd_test), total=len(dataloader_vd_test), desc="test") # checkpoint
    with torch.no_grad():
        for i, batch_data in test_iterator:
            _path, dataV, label = batch_data
            print('Enter batch loader')
            print(dataV.shape)
            #past_data = get_vector_vd(dataV)
            past_data = get_vector_vd(dataV,layer,model_df)
            past_data = past_data.to(args.device)
            label = label.to(args.device)
            loss, prediction = model(past_data,label)
            print("prediction is {}".format(prediction))



if __name__ == '__main__':
    main()


#%%

"""# model test"""
