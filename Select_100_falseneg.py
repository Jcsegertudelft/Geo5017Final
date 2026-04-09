from ultralytics import YOLO
import os
from shutil import copyfile

def main(model_path, image_dir, label_dir,outdir):
    model = YOLO(model_path)
    images = os.listdir(image_dir)
    i = 0

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for img in images:
        im_path = os.path.join(image_dir,img)
        result = model(im_path)[0]
        has_detection = len(result.boxes.conf)>0
        if not has_detection:
            label_name = img.replace('.jpg','.txt')
            label_path = os.path.join(label_dir,label_name)
            with open(label_path,'r') as f:
                lines = f.readlines()
                has_true_detection = len(lines)>0
            if has_true_detection:
                out_path = os.path.join(outdir,img)
                copyfile(im_path,out_path)
                i+=1
        if i == 100:
            break

if __name__ == '__main__':
    main(
        model_path='trained_model_e100.pt',
        image_dir='images/val',
        label_dir='labels/val',
        outdir='false_negatives'
    )