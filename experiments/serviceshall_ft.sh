cd ./src
if $1 == 'default'
then
  nohup python train.py  --exp_id serhall_ft_crowdhuman_dla34_fix --load_model '../models/crowdhuman_dla34.pth' --num_epochs 50 --lr_step '15' --data_cfg '../src/lib/cfg/serhall.json' --K 500 &
  echo 'default'
else
  nohup python train.py  --exp_id serhall_ft_crowdhuman_dla34_fix_max_lr --load_model '../models/crowdhuman_dla34.pth' --num_epochs 50 --lr 5e-4 --data_cfg '../src/lib/cfg/serhall.json' --K 500 &
  echo "max_lr"
fi
cd ..
