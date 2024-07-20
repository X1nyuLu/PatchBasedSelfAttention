#!/bin/bash/

cls_file=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/code/utils/analyze/classifyMoles.py

tgt_path=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/2data_aug/sameTrainDataset/noise/alpha_5/train2/translate/epoch_17/tgt.txt
inference_path=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/FinalResult/2data_aug/sameTrainDataset/noise/alpha_5/train2/translate/epoch_17/BeamSearch_BW10_NB10_result.txt
save_path=$PWD
python ${cls_file} --tgt_path ${tgt_path} --inference_path ${inference_path} --save_path ${save_path}