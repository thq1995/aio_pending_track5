

# python train_dual.py \
# --workers 8 \
# --device 0 \
# --batch 8 \
# --data ../data/crop_yolo_data/data.yaml \
# --img 640 \
# --cfg models/detect/yolov9-e.yaml \
# --weights ../weights/yolov9-e.pt \
# --name yolov9-e \
# --hyp hyp.scratch-high.yaml \
# --min-items 0 \
# --epochs 40 \
# --close-mosaic 15

python -m torch.distributed.launch \
--nproc_per_node 2 \
--master_port 9527 \
train_dual.py \
--workers 8 \
--device 0,1 \
--sync-bn \
--batch 32 \
--data ../data/crop_yolo_data_albumentations/data.yaml \
--img 640 \
--cfg models/detect/yolov9-e.yaml \
--weights ../weights/yolov9-e.pt \
--name yolov9-e-albumentations \
--hyp hyp.scratch-high.yaml \
--min-items 0 \
--epochs 100 \
--close-mosaic 15

## remse_checkpoint