import cv2
import os
# import random
from tqdm import tqdm

# Define class id to label mapping
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

# # Generate a color map for each class
# class_colors = {}
# for class_id in class_labels.keys():
#     class_colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Define specific colors for each class
class_colors = {   # OpenCV uses BGR image format (not RGB as normal)         
    1: (255, 0, 0),     # Blue                - motorbike
    2: (0, 128, 0),     # Green               - DHelmet  
    3: (60, 20, 220),   # Crimson             - DNoHelmet
    4: (30, 105, 210),  # Chocolate           - P1Helmet 
    5: (133, 21, 199),  # Medium Violet Red   - P1NoHelmet
    6: (139, 139, 0),   # Dark Cyan           - P2Helmet
    8: (0, 140, 255),   # Dark Orange         - P0Helmet
    9: (144, 128, 112)  # Slate Gray          - P0NoHelmet
}

# Read ground truth annotations
def read_annotations(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            video_id = int(parts[0])
            frame = int(parts[1])
            left = int(parts[2])
            top = int(parts[3])
            width = int(parts[4])
            height = int(parts[5])
            class_id = int(parts[6])
            confidence = float(parts[7]) 
            if video_id not in annotations:
                annotations[video_id] = {}
            if frame not in annotations[video_id]:
                annotations[video_id][frame] = []
            annotations[video_id][frame].append((left, top, width, height, class_id, confidence))
    return annotations

# annotations = 
# {
#     1: -- Video 1
#     {
#         1: [(left1, top1, width1, height1, class_id1), (left2, top2, width2, height2, class_id2)],
#         2: [(left3, top3, width3, height3, class_id3)]
#     },
#     2: -- Video 2
#     {
#         1: [(left4, top4, width4, height4, class_id4)]
#     },
#     ...
# }

# Visualize video with bounding boxes and labels
def visualize_video(video_path, annotations, output_dir, video_name):
    cap = cv2.VideoCapture(video_path) # Read video
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # This property is used to get the width of the frames in the video stream.  The measuring unit is in pixels. 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # This property is used to get the height of the frames in the video stream. The measuring unit is in pixels.
    fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS stands for frames per second. This property is used to get the frame rate of the video.
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # This property is used to calculate the total number of frames in the video file.

    video_id = int(video_name)
    # video_id = os.path.splitext(os.path.basename(video_path))[0] 
    #     # Extract video_id from the file name
    #     # os.path.basename(video_path) extracts the filename from the full path: 'path/to/video/video.mp4' >> 'video.mp4'.
    #     # os.path.splitext(...) splits the filename into a tuple (root, extension): ('video.mp4') >> ('video', '.mp4')
    output_video_path = os.path.join(output_dir, f'{video_name}_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # FourCC is a 4-byte code used to specify the video codec. The list of available codes can be found in fourcc.org. It is platform dependent.
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened(): # Create a loop that continuously reads frames from the video file until the video capture object is closed
        ret, frame = cap.read()
        # ret is a boolean variable that returns true if the frame is available.
        # frame is an image array vector captured based on the default frames per second defined explicitly or implicitly

        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # It represents the property identifier for the current frame number.

        # video_id = int(cap.get(cv2.CAP_PROP_POS_MSEC)) // (100 * 20) + 1  
        # # cv2.CAP_PROP_POS_MSEC: This property is used to find what is the current position of the video, its measuring unit is milliseconds
        # # Assuming 100 videos each 20 seconds

        # if video_id in annotations and frame_id in annotations[video_id]:
        #     for bbox in annotations[video_id][frame_id]:
        #         left, top, width, height, class_id = bbox
        #         cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        #         label = class_labels.get(class_id, "Unknown")
        #         cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        if frame_id in annotations.get(video_id, {}):
            for bbox in annotations[video_id][frame_id]:
                left, top, width, height, class_id, confidence = bbox
                print(bbox)
                color = class_colors.get(class_id , (0, 255, 0))
                cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)
                label = f"{class_labels.get(class_id , 'Unknown')}: {confidence:.2f}"  # Format label with confidence
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # top - 10 represents the y-coordinate. Subtracting 10 from the top coordinate moves the text slightly above the bounding box of the detected object.
                    # cv2.FONT_HERSHEY_SIMPLEX is one of the built-in font types provided by OpenCV.
                    # 0.5 means the text will be half the size of the base font (number is 1).

        out.write(frame)

    cap.release() #  function call is used to release the video capture object and free up any resources associated with it
    out.release()
    # cv2.destroyAllWindows() # function is used to close all OpenCV windows that are currently open

# Main function
def main(annotations_path, video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
        # When exist_ok is set to True, if the directory already exists,
        # os.makedirs will not raise an error and will proceed without attempting to create the directory again.
    annotations = read_annotations(annotations_path)

    for filename in tqdm(sorted(os.listdir(video_dir))):
        if filename.endswith(".mp4"):
            video_path = os.path.join(video_dir, filename)
            video_name = os.path.splitext(filename)[0]
            if video_name == '048':
                # os.path.splitext(...) splits the filename into a tuple (root, extension): ('video.mp4') >> ('video', '.mp4')
                visualize_video(video_path, annotations, output_dir, video_name)
        
        # break # Test for only 1 video

if __name__ == "__main__":
    # Code to be executed when the script is run as the main program
    # This code will not be executed if the script is imported as a module
    annotations_path = '/home/paperspace/Desktop/aio_pending_track5/data/final_output/video48/_gt_48.txt' 
    video_dir = '/home/paperspace/Desktop/aio_pending_track5/data/aicity2024_track5_test/videos'
    output_dir = '/home/paperspace/Desktop/aio_pending_track5/Visualization/EDA_video'
    main(annotations_path, video_dir, output_dir)
