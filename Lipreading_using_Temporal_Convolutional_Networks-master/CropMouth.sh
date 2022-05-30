python preprocessing/crop_mouth_from_video.py \
--video-direc ../sample/ \
--landmark-direc ./landmarks/LRW_landmarks/ \
--filename-path ../sample/AFTERNOON_detected_face.csv \
--save-direc ./datasets/visual_data/ \
--mean-face ./preprocessing/20words_mean_face.npy \
--convert-gray