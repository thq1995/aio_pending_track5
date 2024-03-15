import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import torch

# Setup logger
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# Define COCO categories for easier reference
COCO_CATEGORIES = {
    'person': 0,
    'motorbike': 3,
}

def is_person_on_motorbike(person_box, motorbike_box, threshold=2.0):
    person_bottom = person_box[3]
    if motorbike_box[1] < person_bottom < motorbike_box[3]:
        person_center = (person_box[0] + person_box[2]) / 2
        motorbike_center = (motorbike_box[0] + motorbike_box[2]) / 2
        if abs(person_center - motorbike_center) < (motorbike_box[2] - motorbike_box[0]) * threshold:
            return True
    return False

def process_image(image_path, output_dir):
    img = cv2.imread(image_path)
    outputs = predictor(img)

    instances = outputs["instances"].to("cpu")
    classes = instances.pred_classes.numpy()
    boxes = instances.pred_boxes.tensor.numpy()

    motorbike_indices = [i for i, cls in enumerate(classes) if cls == COCO_CATEGORIES['motorbike']]
    person_indices = [i for i, cls in enumerate(classes) if cls == COCO_CATEGORIES['person']]

    motorbike_boxes = boxes[motorbike_indices]
    associated_riders = []

    for person_idx, person_box in zip(person_indices, boxes[person_indices]):
        for motorbike_box in motorbike_boxes:
            if is_person_on_motorbike(person_box, motorbike_box):
                associated_riders.append(person_idx)
                break

    # Visualize results
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"][motorbike_indices + associated_riders].to("cpu"))
    result_img = v.get_image()[:, :, ::-1]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, result_img)

def process_directory(directory_path, output_dir):
    for image_name in sorted(os.listdir(directory_path)):
        if image_name.lower().endswith('.jpg'):
            image_path = os.path.join(directory_path, image_name)
            process_image(image_path, output_dir)
            print(f"Processed {image_path}")

if __name__ == '__main__':
    test_images_dir = '/mnt/AI_Data/Development/aio_pending_track5/data/crop_train_frame/images'
    results_dir = '/mnt/AI_Data/Development/aio_pending_track5/data/crop_train_frame/train_detectron_results'
    # image_paths = [os.path.join(test_images_dir, image) for image in sorted(os.listdir(test_images_dir)) if image.endswith('.jpg')]
    process_directory(test_images_dir, results_dir)

