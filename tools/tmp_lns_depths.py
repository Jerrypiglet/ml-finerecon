from pathlib import Path
from tqdm import tqdm
import pickle
import cv2
import png
import numpy as np

SCANNET_DEST = Path('/home/rzhu/Documents/data/finerecon_data/scannet_depths')
SCANNET_DEST.mkdir(exist_ok=True, parents=True)
SCANNET_LISTS = Path('/home/rzhu/Documents/Projects/ScanNet/Tasks/Benchmark/')

# for split in ['train', 'val', 'test']:
for split in ['test']:
    scannet_list_path = SCANNET_LISTS / f'scannetv2_{split}.txt'

    with open(scannet_list_path) as f:
        scene_names = [line.strip() for line in f.readlines()]
        
    for scene_name in tqdm(scene_names):
        if split in ['train', 'val']:
            scene_path = Path('/home/rzhu/Documents/data/simplerecon_data/outputs/HERO_MODEL_scannet_trainval/scannet/default/depths') / scene_name
        else:
            scene_path = Path('/home/rzhu/Documents/data/simplerecon_data/outputs/HERO_MODEL_scannet_test/scannet/default/depths') / scene_name
        assert scene_path.exists(), f'{scene_path} does not exist'
        
        new_scene_path = SCANNET_DEST / scene_name
        if not new_scene_path.exists():
            new_scene_path.symlink_to(scene_path)
            print(f"symlinked {scene_path} to {new_scene_path}")
        
        intrinsic_depth_path = Path('/home/rzhu/Documents/data/finerecon_data/scannet_extracted') / scene_name / 'intrinsic_depth.txt'
        intrinsic_depth_dest = SCANNET_DEST / scene_name / 'intrinsic_depth.txt'
        if not intrinsic_depth_dest.exists():
            intrinsic_depth_dest.symlink_to(intrinsic_depth_path)
            print(f"symlinked {intrinsic_depth_path} to {intrinsic_depth_dest}")
            
        depth_est_path_list = scene_path.glob('*.pickle')
        for _, depth_est_path in enumerate(depth_est_path_list):
            frame_id = int(depth_est_path.stem)
            with open(depth_est_path, 'rb') as f:
                depth_est_dict = pickle.load(f)
            depth_est = depth_est_dict['depth_pred_s0_b1hw'].cpu().numpy().squeeze()
            depth_est_png_path = new_scene_path / 'depth'/ f'{frame_id}.png'
            if not depth_est_png_path.parent.exists():
                depth_est_png_path.parent.mkdir(exist_ok=True, parents=True)
            # cv2.imwrite(str(depth_est_png_path), depth_est)
            
            writer = png.Writer(width=depth_est.shape[1], height=depth_est.shape[0], bitdepth=16)
            depth_est = (depth_est*1000.).astype(np.uint16)
            depth_est = depth_est.reshape(-1, depth_est.shape[1]).tolist()
            with open(str(depth_est_png_path), 'wb') as f: # write 16-bit
                writer.write(f, depth_est)

            print(f"saved {depth_est_png_path}")
            
            # depth_est_path.unlink()
            # print(f"removed {depth_est_path}")

            
