from pathlib import Path
from tqdm import tqdm

SCANNET_ROOT = Path('/home/rzhu/Documents/data/finerecon_data/scannet_extracted')
SCANNET_LISTS = Path('/home/rzhu/Documents/Projects/ScanNet/Tasks/Benchmark/')

for split in ['train', 'val', 'test']:
    scannet_list_path = SCANNET_LISTS / f'scannetv2_{split}.txt'

    with open(scannet_list_path) as f:
        scene_names = [line.strip() for line in f.readlines()]
        
    for scene_name in tqdm(scene_names):
        if split in ['train', 'val']:
            scene_path = SCANNET_ROOT / 'scans' / scene_name
        else:
            scene_path = SCANNET_ROOT / 'scans_test' / scene_name
        assert scene_path.exists(), f'{scene_path} does not exist'
        
        new_scene_path = SCANNET_ROOT / scene_name
        new_scene_path.symlink_to(scene_path)
        
        print(f"symlinked {scene_path} to {new_scene_path}")