
python CenterNet/centernet_detection.py /home/ubuntu/DisEstimation/CenterNet-pytorch-lightning/res /home/ubuntu/DisEstimation/CenterNet-pytorch-lightning/res/annotations --gpus=1 --max_epochs 17 \
--resume_from_checkpoint /home/ubuntu/DisEstimation/CenterNet-pytorch-lightning/chkpts/dla_34-detection-epoch=10-val_loss=2.66.ckpt