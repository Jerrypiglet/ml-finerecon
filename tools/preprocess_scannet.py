'''
To proprocess raw extracted ScanNet data into a format that can be used by FineRecon.

Specifically, new files to be generated:

- intrinsic_color.txt
- intrinsic_depth.txt
- pose.npy

The dataset structure expected by FineRecon is

```
/path/to/dataset/
    test.txt
    train.txt
    val.txt
    first_scan/
        color/
            0.jpg
            1.jpg
            2.jpg
            ...
        depth/
            0.png
            1.png
            2.png
            ...
        intrinsic_color.txt
        intrinsic_depth.txt
        pose.npy
    second_scan/
    ...
    last_scan/
```

Rui Zhu, 2023-08-27

'''

from pathlib import Path
from tqdm import tqdm
import shutil
import numpy as np

SCANNET_ROOT = Path('/newfoundland/ScanNet/extracted')
SCANNET_LISTS = Path('/home/ruizhu/Documents/Projects/ScanNet/Tasks/Benchmark/')

for split in ['train', 'val', 'test']:
    scannet_list_path = SCANNET_LISTS / f'scannetv2_{split}.txt'
    assert scannet_list_path.exists(), f'{scannet_list_path} does not exist'
    with open(scannet_list_path) as f:
        scene_names = [line.strip() for line in f.readlines()]
    
    dest_path = SCANNET_ROOT / ('%s.txt'%split)
    shutil.copyfile(str(scannet_list_path), str(dest_path))
        
    print('Processing %d frames for split: %s'%(len(scene_names), split))
        
    for scene_name in tqdm(scene_names):
        scene_path = SCANNET_ROOT / scene_name
        assert scene_path.exists(), f'{scene_path} does not exist'
        frame_ids = sorted([int(frame_path.stem) for frame_path in scene_path.glob('color/*.jpg')])
        frame_ids_ = sorted([int(frame_path.stem) for frame_path in scene_path.glob('depth/*.png')])
        frame_ids__ = sorted([int(frame_path.stem) for frame_path in scene_path.glob('pose/*.txt')])
        assert frame_ids == frame_ids_ == frame_ids__, f'{scene_path} has inconsistent frame ids'
        
        for intrinsic_file_name in ['intrinsic_color.txt', 'intrinsic_depth.txt']:
            src_path = scene_path / 'intrinsic' / intrinsic_file_name
            dest_path = scene_path / intrinsic_file_name
            shutil.copyfile(str(src_path), str(dest_path))
        
        pose_list = []
        for frame_id in frame_ids:
            pose_path = scene_path / 'pose' / f'{frame_id}.txt'
            P = np.loadtxt(pose_path)
            assert P.shape == (4, 4), f'{pose_path} has invalid shape: {P.shape}'
            P[~np.isfinite(P)] = np.inf
            pose_list.append(P)

            # if not np.all(np.isfinite(P)):  # skip invalid poses
            #     continue
        
        np.save(scene_path / 'pose.npy', np.stack(pose_list))
            
    
            
            
        
        