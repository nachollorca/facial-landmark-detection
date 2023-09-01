# -*- coding: utf-8 -*-
"""data_loading.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aoh_Lr1zJ_d85KFJRcnwwWJW1gi2SDLG

# Script Description: data_loading.py

This script generates the corresponding dataloaders needed by the train.py and test.py scripts to train and evaluate the model respectively, from the corresponding input training.csv file, whose relative path (with respect to the directory where this script is located) is specified in the config.ini file with the variable *'training_csv_path_relative_to_scripts_directory'* (located inside the /app/appplication_files folder). **THIS VARIABLE NEEDS TO BE SET BEFORE THE EXECUTION OF THE SCRIPT. OTHERWISE, THE EXECUTION WILL FAIL**. 

Depending on the value of the *'use_validation_model'* option set in the config.ini file , the behaviour of the script will be different:

> *  *'use_validation_model'* option set to ***True***:

The script drops the rows that contains null values from the input training.csv file, splits the data according to the *'TEST_SPLIT'* constant value set in the config.ini file and creates both training and test datasets (applying augmentation techniques just for the training one to compensate the dropped samples). Finally, their corresponding dataloaders are generated and save in /app/application_file/data_loaders folder as training_loader.pth and test_loader.pth files using as batch size the *'BATCH_SIZE'* constant value set in the config.ini file. 

> *  *'use_validation_model'* option set to ***False***:

The script again drops the rows that contains null values from the input training.csv file, but it does NOT split the data now and just creates one dataset, the training one (applying augmentation techniques as well to compensate the dropped samples). Finally, the corresponding dataloader is generated and save in /app/application_files/data_loaders folder as all_data_loader.pth file using as batch size the *'BATCH_SIZE'* constant value set in the config.ini file. 


---
**NOTE**: All the code implemented to create our customized FaceKeypointDataset and carry out the corresponding augmentation transformation techniques (including brief description. More in detail in paper) can be found in our independent own-built library called customized_dataset_augmentation_library located in the /app/application_files folder.

## Importing required libraries
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import random
import os
import sys
import configparser

print('\n--> data_loading.py execution starts...\n')

"""## Initializing config parser and reading config.ini file"""

config = configparser.ConfigParser()
config.read('./app/application_files/config.ini')

"""## Loading paths, constants and options from configuration file"""

training_csv_path = config['PATHS']['training_csv_path_relative_to_scripts_directory']

BATCH_SIZE = int(config['CONSTANTS']['BATCH_SIZE'])
# If TEST_SPLIT = 0.2, then 80% training, 20% test
TEST_SPLIT = float(config['CONSTANTS']['TEST_SPLIT'])

# Size of images
SIZE = int(config['CONSTANTS']['SIZE'])

# True if we want to reproduce the same results setting fixed seeds
reproducibility = True if config['OPTIONS']['reproducibility'] == "True" else False

# If use_validation_model=True, the training csv specified as training dataset is splited into a training and a test dataset (including augmentation for the traning one), 
# generating their corresponding datalaoders the directory app/application_files/dataloaders. This dataloaders will be used by the corresponding train.py and test.py scripts
# to train and validate the validation model respectively.
# If use_validation_model=False, the training csv specified as training dataset is not splited (but augmentation is applied), being used entirely for training, and
# generating an all_data_loader dataloader file in the directory app/application_files/dataloaders which will be used by the train.py script to train the final model.
use_validation_model = True if config['OPTIONS']['use_validation_model'] == "True" else False

"""## Importing own-built library where our customised FacialKeyPointDataset class is defined as well as all the augmenation operations/transformations applied over the dataset (if the dataset augmentation argument is activated)"""

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

"""# 1) Data loading

## 1.1) Dataframes load: Train_test split and dropping Nan values

### Auxiliary train_test_split function
"""

# train_test_split function
def train_test_split(csv_path, split):
    '''
    This function loads the training data set and splits the dataframe into 
    training_samples and valid_samples. 
    output: training_samples, valid_samples (both panda dataframes)
    '''
    df_data = pd.read_csv(csv_path)
    print('Number samples before droping rows with missing values:', len(df_data))
    # drop all rows with missing values
    df_data = df_data.dropna()
    print('Number samples after droping rows with missing values:', len(df_data))

    training_size = np.ceil(0.8 * len(df_data)).astype('int')
    indices = list(range(len(df_data)))

    # split the data into training and validation:
    df_train = df_data.iloc[indices[:training_size]]
    df_test = df_data.iloc[indices[training_size:]]

    print(f"Training sample instances: {len(df_train)}")
    print(f"Validation sample instances: {len(df_test)}")
    return df_train, df_test

"""### Execution"""

if use_validation_model:
  #train and test dataframe split
  df_train, df_test = train_test_split(training_csv_path, TEST_SPLIT)
else:
  #dataframe without split used for training final model
  df_without_split = pd.read_csv(training_csv_path).dropna()
  print(f"Total sample instances: {len(df_without_split)}")

"""## 1.2) Initialising datasets (applying augmentation to the ones used for training)

### Auxiliary Info Summary After Augmentation function
"""

def info_summary_after_augmentation(df_train, df_test, train_data, valid_data):
  print('\nTraining and test dataset info:')
  print('Original training size:', len(df_train))
  print('Adding our augmented images:', len(train_data))

  print('''The test data size hasn't changed: before''', len(df_test), 'and now', len(valid_data))

  # Let's have a look at the first three images and their corresponding 
  # transformed images

  plt.figure(figsize=(18, 6))
  plt.suptitle('The first images in our training data (before shuffeling)', fontsize=16)
  plt_numbers = [1,2,7,8, 3,4,9,10, 5,6,11,12]

  for i in range(12):
    plt.subplot(2,6,plt_numbers[i])
    plt.imshow(train_data[i]['image'].reshape(SIZE, SIZE), cmap='gray')
    plt.plot(train_data[i]['keypoints'][:,0], train_data[i]['keypoints'][:,1], 'r.')

  plt.show()

"""### Execution"""

if use_validation_model:
  # Initialize both training and test datasets - `FaceKeypointDataset()` used for training validation_model (train) and its corresponding evaluation (test)
  # FaceKeypointDataset class imported from customized_dataset_augmentation_library 
  train_data = customized_dataset_augmentation_library.FaceKeypointDataset(df_train, augmentation=True)
  valid_data = customized_dataset_augmentation_library.FaceKeypointDataset(df_test, augmentation=False)

  print('\nTraining and test datasets created with split included and augmentation applied just for training dataset')

  # Printing info summary after augmentation
  info_summary_after_augmentation(df_train, df_test, train_data, valid_data)
else:
  #Initialize dataset without split applying augmentation for training final model
  all_data = customized_dataset_augmentation_library.FaceKeypointDataset(df_without_split, augmentation=True)

  print('\nTraining dataset created without split and augmentation applied')

  print('\nWhole dataset info for training final model:')
  print('Original size:', len(df_without_split))
  print('Adding our augmented images:', len(all_data))

"""## 1.3) Preparing dataloaders (training ones with shuffle included) and save them in a new app/application_files directory"""

#Create app/application_files/dataloaders directory if doesn't exists
try:
    dirName='app/application_files/dataloaders'
    os.makedirs(dirName)    
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

print('\nBatch size to create the dataloaders: '+ str(BATCH_SIZE))

if use_validation_model:
  # prepare train and test data loaders used for training validation_model (train) and its corresponding evaluation (test)
  train_loader = DataLoader(train_data, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True)
  test_loader = DataLoader(valid_data, 
                            batch_size=BATCH_SIZE, 
                            shuffle=False)

  #Save dataloaders
  torch.save(train_loader, './app/application_files/dataloaders/train_loader.pth')
  torch.save(test_loader, './app/application_files/dataloaders/test_loader.pth')

  print('train_loader.pth and test_loader.pth created and saved in app/application_files/dataloaders folder')
else:
  #Prepare data loader with all data (without split) used for training final model
  all_data_loader = DataLoader(all_data, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)
  #Save dataloader
  torch.save(all_data_loader, './app/application_files/dataloaders/all_data_loader.pth')

  print('all_data_loader.pth created and saved in app/application_files/dataloaders folder')

print('\n--> data_loading.py execution finished')