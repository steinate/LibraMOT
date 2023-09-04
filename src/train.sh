cd /media/zyf301/AE1EF53A1EF4FBE1/pengjq/FairMOT/src
python track.py mot --alltrain_mot17 True --load_model ../exp/mot/ext_offset_near/model_last.pth --conf_thres 0.4 --gpus 1 --ext_offset 'near'
python track.py mot --alltest_mot17 True --load_model ../exp/mot/ext_offset_near/model_last.pth --conf_thres 0.4 --gpus 1 --ext_offset 'near'
cd /media/zyf301/AE1EF53A1EF4FBE1/pengjq/MOTdata/MOT17/images/results/MOT17_test_public_dla34
zip ext_offset_near.zip *.txt

cd /media/zyf301/AE1EF53A1EF4FBE1/pengjq/FairMOT/src
python train.py mot --exp_id ext_offset_max --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot17.json' --gpus 0 --ext_offset 'max'
python track.py mot --alltest_mot17 True --load_model ../exp/mot/ext_offset_max/model_last.pth --conf_thres 0.4 --gpus 1 --ext_offset 'max'
python track.py mot --alltrain_mot17 True --load_model ../exp/mot/ext_offset_max/model_last.pth --conf_thres 0.4 --gpus 1 --ext_offset 'max'

cd /media/zyf301/AE1EF53A1EF4FBE1/pengjq/MOTdata/MOT17/images/results/MOT17_test_public_dla34
zip ext_offset_max.zip *.txt