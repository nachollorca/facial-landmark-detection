## How to run the code
1. We ask the user to run any of the independent python scripts as well as the demo.colab notebook using the cloud-service Google Colab since we have been using this platform as IDE as well as runtime enviroment (taking into account that most of our tests/executions have been carried out there). Moreover, the python evironment provided by Google Colab already includes many packages we have used in the scripts and which are essential to run them. Thus, if the user tries to execute them in an alternative python environment, the execution will fail due to this lack of packages unless the user installs them. 

2. Besides the packages provided by the Google Colab environment, another two external packages (not included by default in the Google Colab's environment) are required to be installed in the python environment which is used to run the scripts: the configparser (needed to deal with the config file) and albumentations (needed to execute augmentation techniques) packages. The user can install them by using pip with the two following commands:
    - !pip install configparser
    - !pip install -U git+https://github.com/albu/albumentations > /dev/null && echo

3. It is strictly necessary to execute the scripts from the submission folder delivered by the project group. The structure of the files and directories inside the submission folder needs to be respected as it stands and cannot suffer any modification. This is because the scripts make use of files inside this submission folder and they access these files by means of relative paths.

*IMPORTANT* Taking into account all these three points, we ask the user to follow these steps to run the scripts:

    1) Upload the whole repo to your Google Drive
    
    2)
        - In case of running the demo.colab notebook:
        
            2.1) Edit the demo.colab at the beginning of the notebook in the 'Setting Runtime Environment' section and change the indicated line of this code where it changes the current directory to the the submission folder using the corresponding path (it depends on the place where the user has stored the submission folder in his/her Google Drive):
            
                #Install configparser external library
                !pip install configparser 
                #Install albumentantions external library
                !pip install -U git+https://github.com/albu/albumentations > /dev/null && echo 
                
                #External imported method required for mounting google drive
                from google.colab import drive 
                #Mouting drive
                drive.mount('/content/drive') 
                
                # HERE!! Please adapt the following line so that the path references the submisson folder (it depends on where
                # you have stored the submission folder in your Google Drive)
                %cd drive/MyDrive/DL Group Project/Project_Code/Submission
                
                import configparser
                config = configparser.ConfigParser()
                config.read('./app/application_files/config.ini')
                
        - In case of running any of the python scripts (data_loading.py, train.py or test.py) individually:
        
            * In case of running the data_loading.py script, first of all upload your training.csv file to your Google Drive (if not done previously) and update the training_csv_path_relative_to_scripts_directory variable value in the config.ini file with its corresponding relative path with respect to the data_loading.py script's location (i.e inside the submission folder's location).
            
            2.1) Create a notebook inside the submission folder
            
            2.2) Edit this notebook and add the following code changing the line of this code where it changes the current directory to the the submission folder using the corresponding path (it depends on the place where the user has stored the submission folder in his/her Google Drive) as well as the last line to decide which python script is to be run:
            
                #Install configparser external library
                !pip install configparser 
                #Install albumentantions external library
                !pip install -U git+https://github.com/albu/albumentations > /dev/null && echo 
                
                #External imported method required for mounting google drive
                from google.colab import drive 
                #Mouting drive
                drive.mount('/content/drive') 
                
                # HERE!! Please adapt the following line so that the path references the submisson folder (it depends on where
                # you have stored the submission folder in your Google Drive)
                %cd drive/MyDrive/DL Group Project/Project_Code/Submission
                
                # HERE!! CHANGE THE PYTHON SCRIPT TO BE EXCUTED!! In this case, the data_loading.py script will be executed.
                execfile('data_loading.py')
            
    3) Run the notebook (depending on the case, the demo.colab or the one just created) using Google Colab.
    
NOTE: Actual behaviour of the scripts in runtime with the default configuration as it stands with the current values in config.ini file:

    - demo.colab: It justs loads the already trained model 'final_model.pth' (trained using the whole dataset - no split) from /models/VGG folder since it is the model architecture selected by default in the config.ini file. Then it takes 9 random photos from the test.csv file located in /app/application_files folder and makes prediction over them using the loaded model, and shows the results. Finally, it runs the code to turn on the user's webcam and starts making predictions (with the loaded model) in real-time using the video stream fetched from the webcam.
    - data_loading.py: Given that the use_validation_model option is set to False in config.ini file, the script takes the training.csv file whose relative path is specified in the config.ini file with the variable training_csv_path_relative_to_scripts_directory (REMEMBER TO CHANGE IT BEFORE ITS EXECUTION!), drops the rows that contains null values and creates the corresponding dataset (applying augmentation techniques) and its dataloader without doing any split, and saves this last one as all_data_loader.pth in the /app/application_files/dataloaders folder.
    - train.py: Given that the use_validation_model option is set to False in config.ini file, the script loads the all_data_loader.pth dataloader from the /app/application_files/dataloaders folder, and uses it to train the model with the corresponding parameters (model architecture, batch size, learning rate and epochs) set in the config.ini. Eventually it stores the trained model in /models/VGG folder (since it is the model architecture selected by default in the config.ini file) as final_model.pth and stores a plot of the train loss with respect to the epochs in /app/application_files/loss_plots/VGG as final_model_train_loss.png.
    - test.py: The script loads the test_loader.pth dataloader (obtained after executing the data_loading.py with the the option use_validation_model activated) from the /app/application_files/dataloaders folder as well as the validation_model.pth from /model/VGG folder (since it is the model architecture selected by default in the config.ini file and it was obtained after executing the train.py with the the option use_validation_model activated), and evaluates the model using the dataloader, and prints the validation_loss over the whole test dataset as well as an example image (obtained from this test_loader) with their corresponding predictions.

NOTE: All the corresponding validation_model.pth and final_model.pth models for each of the three available architectures (ResNet, VGG nad MobileNet) have already been generated and stored in their corresponding folders inside the /models folder with the default configuration constant settings (just changing the model architecture for each case) and training.csv file provided in the project for the model training.

NOTE ABOUT CONFIGURATION FILE: Before runnig any of the scripts, you can set up their configuration for the execution in the config.ini file located in the folder /app/application_files as well. This config.ini file is divided into three sections, each one with different options/variables whose values the user can edit:

    -PATHS section:
        * training_csv_path_relative_to_scripts_directory = ../Data/training.csv ## It specifies to the data_loading.py script the relative path with respect to itself where the training.csv file is located. The data_loading.py script will read this file and use its data to generate the corresponding customized datasets and dataloaders needed by the others scripts. Its current value (../Data/training.csv) is the one used by the project group, where we had stored this csv file. !!!WARNING!!! THE USER NEEDS TO CHANGE ITS VALUE BEFOREHAND IN CASE OF RUNNING THE DATA_LOADING.PY SCRIPT, EITHER IF IT IS EXECUTED INDEPENDENTLY OR FROM THE DEMO.COLAB FILE (in demo.colab case if generate_model option is activated).
        
    -CONSTANTS section:
        * MODEL_ARCHITECTURE = VGG # It specifies the model architecture used to build and train the model. Only three possible values [RESNET, VGG , MOBILENET] to select one of the three available architectures.
        * BATCH_SIZE = 256  ## It specifies the batch size used in the model training done in the train.py script, as well for the evaluation in the test.py script.
        * LR = 0.0001 ## It specifies the learning rate in the model training done in the train.py script.
        * EPOCHS=400 ## It specifies the number of epochs used in the model training done in the train.py script.
        * TEST_SPLIT=0.2 ## It specifies percentage (0.2 -> 20% test & 80% training) used in the data_loading.py script to split the input training.csv file and create both training and test datasets and their corresponding dataloaders that both train.py and test.py scripts will use to train and evaluate the model.
        * SIZE=96 ## It specifies the size of the image. Its value is used across all the scripts.
    
    -OPTIONS section:
        * use_validation_model = False #If this option is set to True, the data_loading.py will split the input training.csv file and create both training and test datasets and their corresponding dataloaders (storing these last ones in /app/application_file/data_loaders as training_loader.pth and test_loader.pth). The train.py script will use the training_loader.pth datalaoder in /app/application_file/data_loaders to train the model and saving it inside /models folder inside the coressponding architecture folder (depending on the model architecture set in the config.ini file) as validation_model.pth as well if the option is activated. If it is set to False, the data_loading.py will not split the input training.csv file and just create one dataset and its corresponding dataloader with the whole data (storing this in /app/application_file/data_loaders as all_data_loader.pth). The train.py will use the all_data_loader.pth datalaoder in /app/application_file/data_loaders to train the model and saving it in /models folder inside the corresponding architecture folder (depending on the model architecture set in the config.ini file) as final_model.pth.
        * generate_model = False # If this option is set to True, when we execute the demo.colab file, it will run the data_loading.py script to create the corresponding dataloaders (or overwrite them if they alredy exist), then run the train.py script to train and build the model with the corresponding dataloader and saving it in /models folder inside the corresponding architecture folder (depending on the model architecture set in the config.ini file), and, if the use_validation_model is activated, run the test.py script using the test dataloader to evaluate the validation model.
        * show_images_during_training = True #If the option is set to True, the train.py script plots an example image containing their labels and predictions of the model for different epochs so we can see how well the model predicts in each of these epochs. Useful to see the training evolution. Otherwise, no image plot.
        * reproducibility = True #If the option is set to True, it forces reproducibility in the scripts data_loading.py, train.py and test.py.

   
## RESOURCES used according to copyright rules
- Base code structure for the data_loading.py (including preprocessing), train.py and test.py scripts
    * https://debuggercafe.com/getting-started-with-facial-keypoint-detection-using-pytorch/
- ResNet model Architecture implementation:
    * http://d2l.ai/chapter_convolutional-modern/resnet.html
- Webcam feature to make predictions using a model in real-time using the video stream fetched from the webcam.
    * https://github.com/theAIGuysCode/colab-webcam
- Face detector model implementation used in the webcam feature for face recognition:
    * https://github.com/opencv/opencv/tree/master/samples/dnn
- VGG model Architecture implementation:
    * https://www.kaggle.com/chr9843/myfacialkeypointsnb
