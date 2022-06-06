# python preprocessing/crop_mouth_from_video.py \
# --video-direc ../sample/ \
# --video-format .mp4 \
# --landmark-direc ./landmarks/LRW_landmarks/ \
# --filename-path ../sample/AFTERNOON_detected_face.csv \
# --save-direc ./datasets/visual_data/ \
# --mean-face ./preprocessing/20words_mean_face.npy \
# --convert-gray

python preprocessing/crop_mouth_from_video.py \
--video-direc ../hangul/ \
--video-format .avi \
--landmark-direc ./landmarks/hangul_landmarks/ \
--filename-path ../hangul/korean_detected_face.csv \
--save-direc ./datasets/visual_data/ \
--mean-face ./preprocessing/20words_mean_face.npy \
--convert-gray