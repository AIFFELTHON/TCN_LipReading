# python get_face_alignment.py \
# --config-path configs/lrw_resnet18_mstcn.json \
# --model-path models/lrw_resnet18_mstcn_adamw_s3.pth.tar \
# --device cuda \
# --queue-length 30 \
# --video-data ../sample/AFTERNOON.mp4 \
# --label-path labels/500WordsSortedList_backup.txt

python get_face_alignment.py \
--config-path configs/lrw_resnet18_mstcn.json \
--model-path models/lrw_resnet18_mstcn_adamw_s3.pth.tar \
--device cuda \
--queue-length 29 \
--video-data ../hangul/오늘.avi \
--label-path labels/500WordsSortedList.txt
