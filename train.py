# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tj0k--0Rv1k0KLOR8k0lZYuHIk9q_RY9

# Script Description: train.py

This script loads the corresponding training dataloader from the /app/application_files/dataloaders folder, and uses it to build and train the model with the corresponding parameters (model architecture, batch size, learning rate and epochs) set in the config.ini. Eventually it stores the trained model in /models folder and stores a plot of the train loss with respect to the epochs in /app/application_files/loss_plots. Both model and plot are stored inside the corresponding model architecture folder according to the one used in the training. E.g. /models/VGG and /app/application_files/loss_plots/VGG folders if the VGG model architecture was set in the config.ini file for the training.

Depending on the value of the *'use_validation_model'* option set in the config.ini file , the behaviour of the script will be different:

> *  *'use_validation_model'* option set to ***True***:

The script loads the training_loader.pth file (generated if data_loading.py script was executed with the *'use_validation_model'* option set to ***True*** in the config.ini file to make the split) as training dataloader from /app/application_file/data_loaders to train the model and, after being trained, it saves the model inside /models folder inside the corresponding architecture folder (depending on the model architecture set in the config.ini file) as **validation_model.pth** file. The training loss plot is also saved as validation_model_train_loss.png in the corresponding folder according to the architecture used.

> *  *'use_validation_model'* option set to ***False***:

The script loads the all_data_loader.pth file (generated if data_loading.py script was executed with the *'use_validation_model'* option set to ***False*** in the config.ini file to avoid making the split) as training dataloader from /app/application_file/data_loaders to train the model and, after being trained, it saves the model inside /models folder inside the corresponding architecture folder (depending on the model architecture set in the config.ini file) as **final_model.pth** file. The training loss plot is also saved as final_model_train_loss.png in the corresponding folder according to the architecture used.

---


The training parameters are set according the values of the following different constant in the config.ini file:

* ***MODEL_ARCHITECTURE*** : This constant specifies the model architecture used to build and train the model from the ones available. Only three possible values [*RESNET, VGG , MOBILENET*] to select one of the three available architectures. 

* ***LR*** : This constant specifies the learning rate to be used in the model training.

* ***EPOCHS*** : This constant specifies the number of epochs to be used in the model training.

## Importing required libraries
"""

import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import torch
import random
import os
import sys
import configparser

print('\n--> train.py execution starts...\n')

"""## Setting GPU as processing device instead of CPU"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""## Initializating config parser and reading config.ini"""

config = configparser.ConfigParser()
config.read('./app/application_files/config.ini')

"""## Loading paths, constants and options from configuration file"""

# learning parameters
LR = float(config['CONSTANTS']['LR'])
EPOCHS = int(config['CONSTANTS']['EPOCHS'])

# Batch size
BATCH_SIZE = int(config['CONSTANTS']['BATCH_SIZE'])

#Model architecture
MODEL_ARCHITECTURE = config['CONSTANTS']['MODEL_ARCHITECTURE']

# Size of images
SIZE = int(config['CONSTANTS']['SIZE'])

# Decide if you want to see an example image during training to visualize the model development:
show_images_during_training = True if config['OPTIONS']['show_images_during_training'] == "True" else False

# True if we want to reproduce the same results setting fixed seeds
reproducibility = True if config['OPTIONS']['reproducibility'] == "True" else False

# If use_validation_model=True, this script will generate a validation_model.pth file (in the models folder) which contains the model trained using the train_loader.pth dataloader file generated by the data_loading.py
# script using the corresponding training data after applying the specified split and applying augmentation. This generated model will be used by the test.py script to evaluate the model over the test dataset.
# If use_validation_model=False, this script will generate a final_model.pth file (in the models folder) which contains the model trained using the all_data_loader.pth dataloader file generated by the data_loading.py
# script using the the whole training data provided in the csv and applying augmentation. This generated final model will be the one used by application to detect the facial landmarks in the video stream.
use_validation_model = True if config['OPTIONS']['use_validation_model'] == "True" else False

"""## Importing own-built architecture library where our model architectures are defined and own-built library where our customised FacialKeyPointDataset class is defined since is required to load the dataloaders"""

import architecture

#Adding path (where the customised library is located) to the system path so it can be imported
library_path = os.getcwd()+'/app/application_files'
sys.path.append(library_path)


import customized_dataset_augmentation_library

"""## Setting fixed seeds and configurations for reproducibility if option activated"""

if reproducibility:
  
  torch.backends.cudnn.deterministic = True
  random.seed(1)
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)
  np.random.seed(1)

"""# 1) Loading Training Dataloader"""

print('Loading training dataloader ...')

if use_validation_model:
  try:
    training_loader = torch.load('./app/application_files/dataloaders/train_loader.pth')
  except FileNotFoundError:
    raise RuntimeError('No train_loader.pth in /app/application_files/dataloaders folder. It has not been generated yet. Run the data_loading.py script with the corresponding config settings to generate it.')
else:
  try:
    training_loader = torch.load('./app/application_files/dataloaders/all_data_loader.pth')
  except FileNotFoundError:
    raise RuntimeError('No all_data_loader.pth in /app/application_files/dataloaders folder. It has not been generated yet. Run the data_loading.py script with the corresponding config settings to generate it.')

print('training dataloader loaded\n')

"""# 2) Training the validation model and final model (this last one if option activated)"""

def train(model, train_dataloader):
    # Set the model in training mode.
    model.train()

    # for each epoch we save the loss of our model over the training data
    train_loss = []

    # calculate the number of batches
    num_batches = int(len(train_dataloader.dataset)/train_dataloader.batch_size)

    for epoch in range(EPOCHS):
        # We will need the following two variables to compute the loss after each epoch
        train_running_loss = 0.0
        counter = 0
    
        for data in train_dataloader:
            counter += 1
            # extract the images and keypoints for the given batch of training data
            images, keypoints = data['image'].to(DEVICE), data['keypoints'].to(DEVICE)
            # flatten the keypoints (original: torch.Size([256, 15, 2]))
            # new size: torch.Size([256, 30])
            keypoints = keypoints.view(keypoints.size(0), -1)

            # set all gradients to zero before using the model
            optimizer.zero_grad()
            outputs = model(images)
            # compute the loss -> single scalar value
            loss = criterion(outputs, keypoints)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()
            #storing last batch and outputs for the "Actual vs Predicted" Keypoints Comparison Plotting
            last_batch=data
            last_batch_outputs=outputs.clone().detach() 

        # We have computed the MSE-loss for each individual batch. In train_running_loss
        # we have the sum over all these MSE-loss, to get the final mean, we have
        # to devide the result by the number of batches (-> counter)
        train_epoch_loss = train_running_loss/counter
        train_loss.append(train_epoch_loss)

        #Getting image, actual keypoints and predicted output keypoints for the first sample of the last batch used for the "Actual vs Predicted" Keypoints Comparison Plotting
        first_sample_image_in_last_train_batch, first_sample_actual_keypoints_in_last_train_batch, first_sample_predicted_keypoints_in_last_train_batch = last_batch['image'][0], last_batch['keypoints'][0], last_batch_outputs[0].cpu()

        if (epoch+1)%25==0 or epoch==0:
          print(f"\nEpoch {epoch+1} of {EPOCHS}")
          print(f"Train Loss: {train_running_loss/counter:.4f}")

          if show_images_during_training:
            print(f'Actual (red) vs Predictive (blue) Keypoints Comparison Plotting for the first sample of the last training batch:')
            plt.clf()# clean plot
            plt.imshow(first_sample_image_in_last_train_batch.reshape(SIZE, SIZE), cmap='gray')
            plt.plot(first_sample_actual_keypoints_in_last_train_batch[:,0], first_sample_actual_keypoints_in_last_train_batch[:,1], 'r.')
            plt.plot(first_sample_predicted_keypoints_in_last_train_batch[::2], first_sample_predicted_keypoints_in_last_train_batch[1::2], 'b.')
            plt.show()

    return train_loss

import torch.nn as nn

#MODEL TRAINING
print('MODEL TRAINING STARTS:\n')

print('- Model architecture used: '+MODEL_ARCHITECTURE+'\n')
print('- Training settings:')
print('  * Learning rate: '+str(LR))
print('  * Epochs: '+ str(EPOCHS) +'\n')

if MODEL_ARCHITECTURE == "RESNET":
  model = architecture.FaceKeypointModelResNet().to(DEVICE)
elif MODEL_ARCHITECTURE == "MOBILENET":
  model = architecture.FaceKeypointModelMobileNet().to(DEVICE)
elif MODEL_ARCHITECTURE == "VGG":
  model = architecture.FaceKeypointModelVGG().to(DEVICE)
else:
  raise RuntimeError('No valid value for model architecture in config.ini file')

# we need a loss function which is good for regression like MSELoss
criterion = nn.MSELoss()
# define optimizer:
# (Adam is an alternative for our classic SGD 
# (https://pytorch.org/docs/stable/optim.html))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_loss = train(model, training_loader)

print('\nMODEL TRAINING DONE')

"""# 3) Saving trained model and loss plot"""

#Create models directory if doesn't exists
try:
    model_dirName='models/'+MODEL_ARCHITECTURE
    os.makedirs(model_dirName)    
    print("Directory " , model_dirName ,  " Created ")
except FileExistsError:
    print("Directory " , model_dirName ,  " already exists")

#Create loss_plots directory if doesn't exists
try:
    loss_plot_dirName='app/application_files/loss_plots/'+MODEL_ARCHITECTURE
    os.makedirs(loss_plot_dirName)    
    print("Directory " , loss_plot_dirName ,  " Created ")
except FileExistsError:
    print("Directory " , loss_plot_dirName ,  " already exists")

#loss plot
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
fig = plt.gcf()
plt.show()

if use_validation_model:
  #VALIDATION MODEL
  #saving loss plot
  fig.savefig(loss_plot_dirName +'/validation_model_train_loss.png')
  #saving model
  torch.save({
              'epoch':EPOCHS,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, model_dirName+'/validation_model.pth')
  
  print('VALIDATION MODEL AND LOSS PLOT SAVED')
else:
  #FINAL MODEL
  #saving loss plot
  fig.savefig(loss_plot_dirName+'/final_model_train_loss.png')
  #saving model
  torch.save({
              'epoch':EPOCHS,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': criterion,
              }, model_dirName+'/final_model.pth')

  print('FINAL MODEL AND LOSS PLOT SAVED')

print('\n--> train.py execution finished')