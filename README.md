# Geo5017Final
Job Segers, Timber Groeneveld, Akhil Veeranki

# Models
The Github contains the pretrained YOLOv8nano model (yolov8n.pt), and the various trained models mentioned in the paper. The number after e indicates the number of epochs, multiscale_xxx indicates that the multiscale parameter was set to multiscale = x.xx

dataset.yaml is used for the model trainer to locate the images and know the classes. The first line may need to be changed to match our local environment.

# Code
Train_model.py is used to train the model, with small alterations based on the which parameters are tested

Plot_results.py is used to visualize the loss functions on the validation dataset

Compare_results.py is used to calculate the general and class-specific image wide precision, recall and F1-score for the validation set, given a certain model

Select_top_100.py copies the top 100 confidence detections of the test set into a new top_100_detections folder. The maximum confidence is first obtained per image, and then the top 100 of these are selected

Select_100_falsepos.py en Select_100_falseneg.py select 100 or all cases of false positives and negatives from the validation set as a sample for error analysis. These will be copied into new folders. Being a false positive or false negative is determined image wide. 

# Requirements
numpy: obtain through pip install numpy

pandas: obtain through pip install pandas

matplotlib: obtain through pip install matplotlib

ultralytics: obtain through pip install ultralytics

Several standard python packages: namely os, statistics and shutil are also utilized. These should be available by default
