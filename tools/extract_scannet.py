from pathlib import Path
import sys
sys.path.insert(0, '/home/ruizhu/Documents/Projects/ContraNeRF')
print(sys.path)
from utils.utils_misc import run_cmd

SCANNET_PATH = Path('/newfoundland/ScanNet/scans')
SCANNET_DUMP_PATH = Path('/newfoundland/ScanNet/extracted')
SCANNET_REPO_PATH = Path('/home/ruizhu/Documents/Projects/ScanNet')
PYTHON2_PATH = Path('/home/ruizhu/miniconda3/envs/scannet_py2/bin/python')
scannet_reader_path = SCANNET_REPO_PATH / 'SensReader/python/reader.py'
assert scannet_reader_path.exists()

    
from multiprocessing import Pool
import time
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
# pick the number according to your CPU cores and memory
parser.add_argument('--worker_total', type=int, default=16, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
parser.add_argument('--if_test', type=bool, default=False, help='switch paths for test split')
opt = parser.parse_args()

if opt.if_test:
    SCANNET_PATH = Path('/newfoundland/ScanNet/scans_test')
    
scene_name_list = [scene_name.stem for scene_name in SCANNET_PATH.iterdir() if 'scene' in scene_name.name]
# scene_name_list = scene_name_list[-2:];

scene_name_list_dump = []

print(f'Number of scenes: {len(scene_name_list)}. Checking dumped scenes...', scene_name_list[:5])

for scene_name in tqdm(scene_name_list):
    if_dumped = True
    scene_path = SCANNET_DUMP_PATH / scene_name
    for folder_name in ['color', 'depth', 'intrinsic', 'pose']:
        subfolder_path = scene_path / folder_name
        if not subfolder_path.exists():
            if_dumped = False
            break
        
    for intrinsic_file_name in ['extrinsic_color.txt', 'extrinsic_depth.txt', 'intrinsic_color.txt', 'intrinsic_depth.txt']:
        intrinsic_file = scene_path / 'intrinsic' / intrinsic_file_name
        if not intrinsic_file.exists():
            if_dumped = False
            break

    frame_ids = sorted([int(frame_path.stem) for frame_path in scene_path.glob('color/*.jpg')])
    frame_ids_ = sorted([int(frame_path.stem) for frame_path in scene_path.glob('depth/*.png')])
    frame_ids__ = sorted([int(frame_path.stem) for frame_path in scene_path.glob('pose/*.txt')])
    if not frame_ids == frame_ids_ == frame_ids__:
        if_dumped = False
        
    if not if_dumped:
        scene_name_list_dump.append(scene_name)
        
print(f'Number of scenes to dump: {len(scene_name_list_dump)}', scene_name_list_dump[:5])
input('Press Enter to continue...')
    
def process_scene(scene_name):
    scene_path = SCANNET_PATH / scene_name
    print(f'{scene_name}: {len(list(scene_path.glob("*.sens")))}')
    sens_path = list(scene_path.glob("*.sens"))[0]
    output_path = SCANNET_DUMP_PATH / scene_name
    output_path.mkdir(exist_ok=True, parents=True)
    
    cmd = '%s %s --filename %s --output_path %s --export_depth_images --export_color_images --export_poses --export_intrinsics'%(str(PYTHON2_PATH), str(scannet_reader_path), str(sens_path), str(output_path))
    # print(cmd)
    _output = run_cmd(cmd)
    # print(_output)

tic = time.time()
p = Pool(processes=opt.worker_total)

list(tqdm(p.imap_unordered(process_scene, scene_name_list_dump), total=len(scene_name_list_dump)))
p.close()
p.join()
print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))

