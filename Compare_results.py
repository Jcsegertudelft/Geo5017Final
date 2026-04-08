import os
import numpy as np
from ultralytics import YOLO
from statistics import harmonic_mean
from pprint import pprint

def precis_and_recall(model_path, image_folder, label_folder):
    model = YOLO(model_path)
    pred_list = []
    true_list = []
    for img in os.listdir(image_folder):
        #Obtain prediction: is there trash in image
        res = model(os.path.join(image_folder,img))[0]
        predicted = len(res.boxes.conf) != 0
        pred_list.append(predicted)

        #Obtain ground truth: is there trash in image
        label_filename = img.replace('.jpg','.txt')
        label_path = os.path.join(label_folder,label_filename)
        with open(label_path,'r') as f:
            true = len(f.readlines()) != 0
        true_list.append(true)

    #Calculate precision, recall and F1_score
    pred_array = np.array(pred_list)
    true_array = np.array(true_list)
    true_pos_count = np.sum(pred_array[true_array])
    false_pos_count = np.sum(pred_array[~true_array])
    false_neg_count = np.sum(true_array[~pred_array])

    precision = true_pos_count/(true_pos_count+false_pos_count)
    recall = true_pos_count/(true_pos_count+false_neg_count)
    f1_score = harmonic_mean([precision, recall])
    return precision, recall, f1_score

if __name__ == "__main__":
    image_dir = 'images/val'
    label_dir = 'labels/val'
    model1_path = 'trained_model_run1.pt'
    model2_path = 'trained_model_run2.pt'
    results_1 = precis_and_recall(model1_path, image_dir, label_dir)
    results_2 = precis_and_recall(model2_path, image_dir, label_dir)
    print(results_1)
    print(results_2)


