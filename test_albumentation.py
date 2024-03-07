import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the transformation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Example image path - replace with your actual image path
image_path = "/path/to/your/image.jpg"


import cv2
import numpy as np
import albumentations as A

# Assuming 'transformed_image', 'transformed_bboxes', and 'transformed_class_labels' are obtained as shown in your snippet

def clamp_normalized_values(value):
    """Clamp the normalized value to the range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))

def adjust_transformed_bboxes(bboxes):
    """
    Adjust bounding boxes to ensure all normalized values are within the [0.0, 1.0] range.
    This is especially important after transformations that might result in slight numerical inaccuracies.
    """
    adjusted_bboxes = []
    for bbox in bboxes:
        center_x, center_y, bbox_width, bbox_height = bbox
        # Clamp the bounding box coordinates to ensure they are within the [0.0, 1.0] range
        center_x = clamp_normalized_values(center_x)
        center_y = clamp_normalized_values(center_y)
        bbox_width = clamp_normalized_values(bbox_width)
        bbox_height = clamp_normalized_values(bbox_height)

        # Further ensure the bounding box does not exceed the boundaries due to width/height exceeding 1
        if center_x + bbox_width / 2 > 1.0:
            bbox_width = (1.0 - center_x) * 2
        if center_y + bbox_height / 2 > 1.0:
            bbox_height = (1.0 - center_y) * 2
        if center_x - bbox_width / 2 < 0.0:
            bbox_width = center_x * 2
        if center_y - bbox_height / 2 < 0.0:
            bbox_height = center_y * 2

        adjusted_bboxes.append([center_x, center_y, bbox_width, bbox_height])

    return adjusted_bboxes


def clamp(value, min_value=0.0, max_value=1.0):
    """Clamp a value to ensure it's within the specified min and max range."""
    return max(min_value, min(max_value, value))

def draw_bboxes_cv2(image, bboxes, class_labels):
    """
    Draw bounding boxes on an image using OpenCV.
    
    Parameters:
    - image: Image as a numpy array.
    - bboxes: Bounding boxes in Albumentations/YOLO format (center_x, center_y, width, height).
    - class_labels: List of class labels for each bounding box.
    """
    img = image.copy()
    # 1080, 1920
    h, w = img.shape[:2]
    
    for bbox, label in zip(bboxes, class_labels):
        # Convert from Albumentations/YOLO format to corner coordinates
        center_x, center_y, bbox_width, bbox_height = bbox
        x1 = int(((center_x - bbox_width / 2)) * w)
        y1 = int(((center_y - bbox_height / 2)) * h)
        x2 = int(((center_x + bbox_width / 2)) * w)
        y2 = int(((center_y + bbox_height / 2)) * h)



        
        print(x1, y1, x2, y2, label)
        
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Optional: Put the class label
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img

def ensure_valid_yolo_bboxes(bboxes, image_width, image_height):
    """
    Ensure bounding boxes are within the image boundaries for YOLO format.
    Assumes bounding boxes are in YOLO format [center_x, center_y, width, height] with normalized coordinates.
    
    Parameters:
    - bboxes: A list of bounding boxes in YOLO format.
    - image_width: Width of the image.
    - image_height: Height of the image.
    
    Returns:
    - A list of adjusted bounding boxes in YOLO format.
    """
    valid_bboxes = []
    for bbox in bboxes:
        center_x, center_y, width, height = bbox
        
        # Convert to absolute coordinates to adjust
        abs_x = center_x * image_width
        abs_y = center_y * image_height
        abs_width = width * image_width
        abs_height = height * image_height
        
        # Ensure the bounding box fits within the image dimensions
        abs_x = max(0, min(abs_x, image_width))
        abs_y = max(0, min(abs_y, image_height))
        abs_width = min(abs_width, image_width - abs_x)
        abs_height = min(abs_height, image_height - abs_y)
        
        # Convert back to normalized YOLO format
        center_x = abs_x / image_width
        center_y = abs_y / image_height
        width = abs_width / image_width
        height = abs_height / image_height

        center_x = clamp(center_x)
        center_y = clamp(center_y)
        width = clamp(width)
        height = clamp(height)

        
        valid_bboxes.append([center_x, center_y, width, height])
        
    return valid_bboxes

# Example image path - replace with your actual image path
image_path = "/home/paperspace/Desktop/aio_pending_track5/data/crop_yolo_data_albumentations/train/images/gt_part_1_000004.jpg"

# Load the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Example bounding boxes in YOLO format [center_x, center_y, width, height]
bboxes = [
    [7.927079999999999682e-01, 2.268500000000000030e-02, 1.197919999999999957e-01, 4.537000000000000061e-02],
    [7.906250000000000222e-01, 5.879600000000000104e-02, 8.645799999999999319e-02, 1.175930000000000031e-01]
]
   
print(bboxes)

bboxes = adjust_transformed_bboxes(bboxes)

print(bboxes)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    # A.RandomCrop(width=600, height=600, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=1, min_visibility=0.1))

# # Example class labels
class_labels = ['DHelmet', 'motorbike']

# # # Apply the transformation
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)

# # Extract transformed items
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']


# image = draw_bboxes_cv2(transformed_image, transformed_bboxes, transformed_class_labels)
image = draw_bboxes_cv2(image=transformed_image, bboxes=transformed_bboxes, class_labels=class_labels)

cv2.imwrite('test.jpg', image)