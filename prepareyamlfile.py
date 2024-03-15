import os
import yaml

yaml_path = 'data/crop_yolo_data/data.yaml'
label_lst = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']

data_dict = {
    'path': os.path.join(os.getcwd(), 'data/crop_yolo_data'),
    'train': 'train/images',
    'val': 'train/images',
    'nc': len(label_lst),
    'names': label_lst
}  

with open(yaml_path, 'w') as f:
    yaml.dump(data_dict, f, sort_keys=False)
