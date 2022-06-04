# python main.py \
# --config-path ./configs/lrw_resnet18_mstcn.json \
# --model-path ./models/lrw_resnet18_mstcn.pth.tar \
# --data-dir ./datasets/visual_data/ \
# --test

python main.py \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./train_logs/tcn_backup/2022-06-04T12:13:35/ckpt.best.pth.tar \
--data-dir ./datasets/visual_data/ \
--label-path ./labels/500WordsSortedList.txt \
--save-dir ./result \
--test
