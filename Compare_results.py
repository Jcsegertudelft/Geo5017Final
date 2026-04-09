import os
import numpy as np
from ultralytics import YOLO
from statistics import harmonic_mean
from pprint import pprint
import pandas as pd

def precis_and_recall(model_path, image_folder, label_folder):
    #Load model
    model = YOLO(model_path)

    #Storage list
    pred_list = []
    true_list = []

    #Storage arrays for class specific stuff
    pred_cls_arr = np.full((5,len(os.listdir(image_folder))), False)
    true_cls_arr = np.full((5,len(os.listdir(image_folder))), False)
    for i,img in enumerate(os.listdir(image_folder)):

        #Obtain prediction: is there trash in image
        res = model(os.path.join(image_folder,img))[0]
        any_predicted = len(res.boxes.conf) != 0
        pred_list.append(any_predicted)

        #If so what kind
        if any_predicted:
            predicted_classes = res.boxes.cls
            for cls in predicted_classes:
                cls = int(cls)
                pred_cls_arr[cls][i]=True



        #Obtain ground truth: is there trash in image
        label_filename = img.replace('.jpg','.txt')
        label_path = os.path.join(label_folder,label_filename)
        with open(label_path,'r') as f:
            lines = f.readlines()
            any_true = len(lines) != 0

            #If so what kind
            if any_true:
                for line in lines:
                    cls = int(line.strip().split(' ')[0])
                    true_cls_arr[cls][i]=True
        true_list.append(any_true)

    #Calculate general precision, recall and F1_score
    pred_array = np.array(pred_list)
    true_array = np.array(true_list)
    true_pos_count = np.sum(pred_array[true_array])
    false_pos_count = np.sum(pred_array[~true_array])
    false_neg_count = np.sum(true_array[~pred_array])

    if true_pos_count == 0 and false_pos_count == 0:
        precision = 0.
    else:
        precision = true_pos_count/(true_pos_count+false_pos_count)

    if true_pos_count == 0 and false_neg_count == 0:
        recall = 0.
    else:
        recall = true_pos_count/(true_pos_count+false_neg_count)
    f1_score = harmonic_mean([precision, recall])

    #Per class precision, recall f1 score
    cls_precision = []
    cls_recall = []
    cls_f1_score = []

    for i in range(5):
        cls_pred = pred_cls_arr[i]
        cls_true = true_cls_arr[i]
        true_pos_count = np.sum(cls_true[cls_pred])
        false_pos_count = np.sum(cls_pred[~cls_true])
        false_neg_count = np.sum(cls_true[~cls_pred])

        if true_pos_count == 0 and false_pos_count == 0:
            cls_precision.append(0.)
        else:
            cls_precision.append(true_pos_count/(true_pos_count+false_pos_count))

        if true_pos_count == 0 and false_neg_count == 0:
            cls_recall.append(0.)
        else:
            cls_recall.append(true_pos_count/(true_pos_count+false_neg_count))
        cls_f1_score.append(harmonic_mean([cls_precision[i], cls_recall[i]]))
    cls_df = pd.DataFrame()
    cls_df['Class_n'] = list(range(5))
    cls_df['Class'] = ['litter','other','bulky waste','cardboard','garbage bag']
    cls_df['Precision'] = cls_precision
    cls_df['Recall'] = cls_recall
    cls_df['F1_score'] = cls_f1_score
    cls_df.to_csv(model_path.replace('.pt','.csv'))

    return precision, recall, f1_score,

if __name__ == "__main__":
    image_dir = 'images/val'
    label_dir = 'labels/val'
    model1_path = 'trained_model_run1.pt'
    model2_path = 'trained_model_run2.pt'
    model3_path = 'trained_model_e100_multi_scale_025.pt'
    #results_1 = precis_and_recall(model1_path, image_dir, label_dir)
    #results_2 = precis_and_recall(model2_path, image_dir, label_dir)
    results_3 = precis_and_recall(model3_path, image_dir, label_dir)
    print(f"precision {results_3[0]:.4f}, recall {results_3[1]:.4f}, f1_score {results_3[2]:.4f}")



