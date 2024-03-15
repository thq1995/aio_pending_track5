import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
from tqdm import tqdm

# Setup logger
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

def process_and_save_info(image_paths, results_dir="results", desired_classes=[0, 3]):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    class_colors = {0: (0, 1, 0), 3: (0, 0, 1)}  # Assign blue to class 0, red to class 3

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        outputs = predictor(img)
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

        if boxes is not None:
            filtered_indices = [i for i, cls in enumerate(classes) if cls in desired_classes]
            filtered_boxes = boxes[filtered_indices]
            filtered_classes = classes[filtered_indices]
            filtered_scores = scores[filtered_indices]
            filtered_masks = masks[filtered_indices]

            # Save data including confidence scores
            data_to_save = [f"{box[0]}, {box[1]}, {box[2]}, {box[3]}, {cls}, {score}\n" for box, cls, score in zip(filtered_boxes, filtered_classes, filtered_scores)]
            txt_filename = os.path.join(results_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
            with open(txt_filename, 'w') as f:
                f.writelines(data_to_save)

            # Visualize with confidence scores
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

            for i, idx in enumerate(filtered_indices):
                box = boxes[idx]
                cls = classes[idx]
                score = scores[idx]
                single_mask = masks[idx]
                color = class_colors[cls]  # Use assigned color based on class
                
                class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", [])[cls]
                label = f"{class_name}: {score:.2f}"
                v.draw_text(label, (box[0], box[1] - 10))
                v.draw_box(box, edge_color=color)
                v.draw_binary_mask(single_mask, color=color)

            visualized_img_path = os.path.join(results_dir, os.path.basename(img_path).replace('.jpg', '_visualized.jpg'))
            cv2.imwrite(visualized_img_path, v.get_output().get_image()[:, :, ::-1])
if __name__ == '__main__':
    test_images_dir = '/mnt/AI_Data/Development/aio_pending_track5/data/crop_train_frame/images'
    results_dir = '/mnt/AI_Data/Development/aio_pending_track5/data/crop_train_frame/train_detectron_results'
    image_paths = [os.path.join(test_images_dir, image) for image in sorted(os.listdir(test_images_dir)) if image.endswith('.jpg')]
    process_and_save_info(image_paths, results_dir)

