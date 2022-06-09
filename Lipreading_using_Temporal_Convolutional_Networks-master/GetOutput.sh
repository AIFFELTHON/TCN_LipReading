python get_output.py \
--config-path configs/lrw_snv1x_tcn2x.json \
--model-path train_logs/tcn/2022-06-08T20:48:29/ckpt.best.pth.tar \
--device cuda \
--queue-length 30 \
--video-data ../hangeul/함께/test/함께_00032.avi \
--label-path labels/500WordsSortedList.txt \
--save-dir ../hangeul/GetOutput_함께
