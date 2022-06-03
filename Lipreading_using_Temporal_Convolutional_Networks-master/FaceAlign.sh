python get_face_alignment.py \
--config-path configs/lrw_resnet18_mstcn.json \
--model-path models/lrw_resnet18_mstcn_adamw_s3.pth.tar \
--device cuda \
--queue-length 30 \
--video-data ../sample/AFTERNOON.mp4
