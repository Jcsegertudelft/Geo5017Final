from ultralytics import YOLO
import os
from shutil import copyfile

def main(model_path, image_dir, label_dir,outdir):
    #Load model and images
    model = YOLO(model_path)
    images = os.listdir(image_dir)
    #Counter
    i = 0

    #Make output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    #For each image no trash detected by model
    for img in images:
        im_path = os.path.join(image_dir,img)
        result = model(im_path)[0]
        has_detection = len(result.boxes.conf)>0
        if not has_detection:
            #Get label path
            label_name = img.replace('.jpg','.txt')
            label_path = os.path.join(label_dir,label_name)
            #For false negative, needs to have a ground truth detection
            with open(label_path,'r') as f:
                lines = f.readlines()
                has_true_detection = len(lines)>0
            #Copy file to output directory
            if has_true_detection:
                out_path = os.path.join(outdir,img)
                copyfile(im_path,out_path)
                i+=1
        #100 images max as a sample of false negatives
        if i == 100:
            break

if __name__ == '__main__':
    main(
        model_path='trained_model_e100.pt',
        image_dir='images/val',
        label_dir='labels/val',
        outdir='false_negatives'
    )