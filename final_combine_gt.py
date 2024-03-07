import os
import re

def sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers]


def main():
    final_path = "/home/paperspace/Desktop/aio_pending_track5/data/final_output"
    final_lines = []
    for video in sorted(os.listdir(final_path), key=sort_key):
        video_path = os.path.join(final_path, video)
        if video_path.endswith(".txt"):
            continue
        for gt_file in os.listdir(video_path):
            if gt_file.endswith(".txt"):
                gt_file_path = os.path.join(video_path, gt_file)
                
                # print(gt_file_path)
                with open(gt_file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        final_lines.append(line)
    
    
    
    submit_file = "/home/paperspace/Desktop/aio_pending_track5/data/final_output/submit.txt"

    with(open(submit_file, "w")) as f:
        for line in final_lines:
        # print(type(line[6]))
            # line[6]=  'test'
            f.write(line)
    
if __name__ == "__main__":
    main()