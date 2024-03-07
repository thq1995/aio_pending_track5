import os
import cv2
from tqdm import tqdm

class_labels = {
    1: "1 - motorbike",
    2: "2 - DHelmet",       # Driver with Helmet
    3: "3 - DNoHelmet",     # Driver without Helmet
    4: "4 - P1Helmet",      # Passenger 1 with Helmet
    5: "5 - P1NoHelmet",    # Passenger 1 without Helmet
    6: "6 - P2Helmet",      # Passenger 2 with Helmet
    7: "7 - P2NoHelmet",    # Passenger 2 without Helmet
    8: "8 - P0Helmet",      # Passenger 0 with Helmet
    9: "9 - P0NoHelmet"     # Passenger 0 without Helmet
}

class_colors = {   # OpenCV uses BGR image format (not RGB as normal)         
    1: (255, 0, 0),     # Blue                - motorbike
    2: (0, 128, 0),     # Green               - DHelmet  
    3: (60, 20, 220),   # Crimson             - DNoHelmet
    4: (30, 105, 210),  # Chocolate           - P1Helmet 
    5: (133, 21, 199),  # Medium Violet Red   - P1NoHelmet
    6: (139, 139, 0),   # Dark Cyan           - P2Helmet
    7: (128, 0, 128),   # Purple              - P2NoHelmet
    8: (0, 140, 255),   # Dark Orange         - P0Helmet
    9: (144, 128, 112)  # Slate Gray          - P0NoHelmet
}


def draw_yolo_bboxes(image, bboxes):

    h, w = image.shape[:2]
    for bbox in bboxes:
        class_id, x_center, y_center, bbox_width, bbox_height = bbox
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)
        color = class_colors.get(class_id + 1, (0, 255, 0))

        class_id = int(class_id)
        # print(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_labels.get(class_id + 1, "Unknown")
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def process_dataset(image_dir, label_dir, output_dir):
  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in tqdm(sorted(os.listdir(image_dir))):
        if image_name.lower().endswith(('.jpg', '.png')):
            # Load the image
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            
            # Load the corresponding label file
            label_path = os.path.join(label_dir, image_name.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    bboxes = [list(map(float, line.split()))[0:] for line in file.readlines()]
                    # print(bboxes)
                # Draw the bounding boxes on the image
                draw_yolo_bboxes(image, bboxes)
            
            # Save the processed image
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(os.path.join(output_dir, image_name), image)

# Example usage
            
            
image_dir = '/home/paperspace/Desktop/aio_pending_track5/data/crop_yolo_data_albumentations/train/images'
label_dir = '/home/paperspace/Desktop/aio_pending_track5/data/crop_yolo_data_albumentations/train/labels'
output_dir = '/home/paperspace/Desktop/aio_pending_track5/EDA/processed_images'
process_dataset(image_dir, label_dir, output_dir)
