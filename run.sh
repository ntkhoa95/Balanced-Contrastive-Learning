# python main.py --data /mnt/data4/taindp/WORKSPACE/04_COMPET/classifier/dataset_phase2_decouple \
#   --dataset mosquito \
#   -b 8 \
#   --use_norm True \
#   --lr 0.001 -p 200 --epochs 100 \
#   --arch convnext_base.fb_in22k_ft_in1k \
#   --wd 1e-4 --cos False \
#   --cl_views sim-sim

python main.py --data /mnt/data4/taindp/WORKSPACE/04_COMPET/classifier/dataset_phase2_decouple \
  --dataset mosquito \
  -b 4 \
  --use_norm True \
  --lr 0.001 -p 200 --epochs 100 \
  --arch maxvit_base_tf_224.in21k \
  --wd 1e-4 --cos False \
  --cl_views sim-sim
