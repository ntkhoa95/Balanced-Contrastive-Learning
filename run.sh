python main.py --data /mnt/data4/taindp/WORKSPACE/04_COMPET/classifier/dataset_phase2_decouple \
  --dataset mosquito \
  -b 8 \
  --use_norm True \
  --lr 0.001 -p 200 --epochs 100 \
  --arch resnet50 \
  --wd 1e-4 --cos False \
  --cl_views sim-sim
