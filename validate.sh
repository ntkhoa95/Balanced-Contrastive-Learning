export CUDA_VISIBLE_DEVICES=1
python validate.py --data /mnt/data4/taindp/WORKSPACE/04_COMPET/classifier/data/preprocess_data \
  --dataset mosquito \
  -b 8 \
  --use_norm True \
  --arch resnext50 \
  --cl_views sim-sim \
  --resume log/mosquito_resnext50_batchsize_8_epochs_100_temp_0.07_lr_0.001_sim-sim/bcl_ckpt.best.pth.tar
