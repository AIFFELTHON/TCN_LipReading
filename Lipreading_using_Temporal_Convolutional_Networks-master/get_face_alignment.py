import argparse
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToTensor, ToPILImage

from lipreading.model import Lipreading
from preprocessing.transform import warp_img, cut_patch

STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]
START_IDX = 48
STOP_IDX = 68
CROP_WIDTH = CROP_HEIGHT = 96


@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def load_model(config_path: Path, num_classes=500):
    with config_path.open() as fp:
        config = json.load(fp)
    tcn_options = {
        'num_layers': config['tcn_num_layers'],
        'kernel_size': config['tcn_kernel_size'],
        'dropout': config['tcn_dropout'],
        'dwpw': config['tcn_dwpw'],
        'width_mult': config['tcn_width_mult'],
    }
    return Lipreading(
        num_classes=num_classes,
        tcn_options=tcn_options,
        backbone_type=config['backbone_type'],
        relu_type=config['relu_type'],
        width_mult=config['width_mult'],
        extract_feats=False,
    )


def visualize_probs(vocab, probs, col_width=4, col_height=300):
    num_classes = len(probs)
    out = np.zeros((col_height, num_classes * col_width + (num_classes - 1), 3), dtype=np.uint8)
    for i, p in enumerate(probs):
        x = (col_width + 1) * i
        cv2.rectangle(out, (x, 0), (x + col_width - 1, round(p * col_height)), (255, 255, 255), 1)
    top = np.argmax(probs)
    cv2.addText(out, f'Prediction: {vocab[top]}', (10, out.shape[0] - 30), 'Arial', color=(255, 255, 255))
    cv2.addText(out, f'Confidence: {probs[top]:.3f}', (10, out.shape[0] - 10), 'Arial', color=(255, 255, 255))
    return out


def main():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(11111111111111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=Path, default=Path('configs/lrw_resnet18_mstcn.json'))
    parser.add_argument('--model-path', type=Path, default=Path('models/lrw_resnet18_mstcn_adamw_s3.pth.tar'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--queue-length', type=int, default=30)
    parser.add_argument('--video-data', type=None, default='sample/AFTERNOON.mp4')
    args = parser.parse_args()
    print(22222222222)

    mean_face_landmarks = np.load(Path('preprocessing/20words_mean_face.npy'))

    with Path('labels/500WordsSortedList.txt').open() as fp:
        vocab = fp.readlines()
    # assert len(vocab) == 500
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=args.device)
    print(33333333333)
    
    model = load_model(args.config_path)
    # model = load_model(args.config_path, num_classes=len(vocab))
    model.load_state_dict(torch.load(Path(args.model_path), map_location=args.device)['model_state_dict'])
    model = model.to(args.device)
    print(44444444)

    queue = deque(maxlen=args.queue_length)
    print(555555555555555)
    
    video_pathname = args.video_data
    cap = cv2.VideoCapture(video_pathname)
    if not cap.isOpened():
        print("could not open : ", video_pathname)
        cap.release()
        exit(0)
    print(6666666666666)

    i = 0
    landmark_idx = 0
    probs_idx = 0
    while True:
        ret, image_np = cap.read()
        if not ret:
            break
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        print(77777777777)

        all_landmarks = fa.get_landmarks(image_np)
        if all_landmarks:
            landmarks = all_landmarks[0]

            # BEGIN PROCESSING

            trans_frame, trans = warp_img(
                landmarks[STABLE_PNTS_IDS, :], mean_face_landmarks[STABLE_PNTS_IDS, :], image_np, STD_SIZE)
            trans_landmarks = trans(landmarks)
            patch = cut_patch(
                trans_frame, trans_landmarks[START_IDX:STOP_IDX], CROP_HEIGHT // 2, CROP_WIDTH // 2)
            print(88888888)
            
            from torchcivion import transforms as transforms
            
            # cv2.imshow('patch', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            PATCH_SAVE_PATH = f'/home/PHR/Lipreading_using_TCN_running/sample/FaceAlign_AFTERNOON/patch/patch_{landmark_idx}.jpg'
            cv2.imwrite(PATCH_SAVE_PATH, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            landmark_idx += 1

            # patch_torch = to_tensor(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)).to(args.device)
            print()
            print()
            print(f'#### 111 patch.shape: {patch.shape}')
            # patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            patch[2] = 1
            print(f'#### 222 patch.shape: {patch.shape}')
            print()
            print()
            patch_torch = to_tensor(patch).to(args.device)
            queue.append(patch_torch)
            print(999999999999)

            if len(queue)+1 >= args.queue_length:
                with torch.no_grad():
                    model_input = torch.stack(list(queue), dim=1).unsqueeze(0)
                    logits = model(model_input, lengths=[args.queue_length])
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probs = probs[0].detach().cpu().numpy()
                
                print(101010101010)

                vis = visualize_probs(vocab, probs)
                # cv2.imshow('probs', vis)
                PROBS_SAVE_PATH = f'/home/PHR/Lipreading_using_TCN_running/sample/FaceAlign_AFTERNOON/probs/probs_{probs_idx}.jpg'
                cv2.imwrite(PROBS_SAVE_PATH, vis)
                probs_idx += 1

            # END PROCESSING

            for x, y in landmarks:
                cv2.circle(image_np, (int(x), int(y)), 2, (0, 0, 255))
            print(000000000)

        # cv2.imshow('camera', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        CAMERA_SAVE_PATH = f'/home/PHR/Lipreading_using_TCN_running/sample/FaceAlign_AFTERNOON/camera/camera_{i}.jpg'
        cv2.imwrite(CAMERA_SAVE_PATH, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key in {27, ord('q')}:  # 27 is Esc
            break
        elif key == ord(' '):
            cv2.waitKey(0)
        
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(123456789)


if __name__ == '__main__':
    main()
