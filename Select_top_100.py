import os
from ultralytics import YOLO
import numpy as np
from shutil import copyfile

def main(mod_path = "trained_model.pt",
         image_dir ='images/test',
         outdir = "top_100_detections"):

    model = YOLO(mod_path)

    image_dir = 'images/test'
    test_images = os.listdir(image_dir)

    max_conf = []
    for img in test_images:
        im_path = os.path.join(image_dir,img)
        result = model(im_path)[0]
        conf = np.array(result.boxes.conf)
        print(conf)
        if len(conf) != 0:
            max_conf.append(float(np.max(conf)))
        else:
            max_conf.append(0.)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    without_dir = os.path.join(outdir, 'without_boxes')
    if not os.path.exists(without_dir):
        os.makedirs(without_dir)

    with_dir = os.path.join(outdir, 'with_boxes')
    if not os.path.exists(with_dir):
        os.makedirs(with_dir)

    indices_100 = np.argpartition(max_conf, -100)[-100:]
    for i in indices_100:
        im_path = os.path.join(image_dir,test_images[i])
        copied_path_no_boxes = os.path.join(without_dir, test_images[i])
        copied_path_with_boxes = os.path.join(with_dir, test_images[i])

        copyfile(im_path, copied_path_no_boxes)
        result = model(im_path)[0]
        result.save(filename=copied_path_with_boxes)


if __name__ == '__main__':
    main()
