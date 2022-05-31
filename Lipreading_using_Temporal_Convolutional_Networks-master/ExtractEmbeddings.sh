python main.py \
--extract-feats \
--data-dir ./datasets/visual_data/ \
--config-path ./configs/lrw_resnet18_mstcn.json \
--model-path ./models/lrw_resnet18_mstcn.pth.tar \
--mouth-patch-path ./datasets/visual_data/AFTERNOON/test/ \
--mouth-embedding-out-path ../sample/embeddings/