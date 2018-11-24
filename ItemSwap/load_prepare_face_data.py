import subprocess
import yaml
from glob import glob
import shutil
import os

urls_path = os.environ.get('urls_path', 'urls_faces.yaml')
print(f'Urls path: {urls_path}')
images_path = os.environ.get('images_path', None)
print(f'Images path: {images_path}')
data_folder_path = os.environ.get('data_folder_path', 'data_faces')
print(f'Data folder path: {data_folder_path}')


# Before: remove all videos, faces, binary_mask

with open(urls_path, 'r') as f:
    urls = yaml.load(f)

print(urls)

# Downloading videos
print('Downloading videos..')
for folder in ['A', 'B']:
    if urls[folder] is None:
        continue
    for i, url in enumerate(urls[folder]):
        print(f'Folder {folder}, url: {url}')
        subprocess.call(
            f'youtube-dl --recode-video mp4 -o {data_folder_path}/videos/{folder}_{i}.mp4 {url}',
            shell=True)

# Small fix of youtube-dl
for p in glob(f'{data_folder_path}/videos/*.mp4.mp4'):
    shutil.move(p, p.replace('.mp4.mp4', '.mp4'))

# Cropping face
print('Cropping faces..')
for folder in ['A', 'B']:
    if urls[folder] is None:
        continue
    for i, url in enumerate(urls[folder]):
        print(f'Folder {folder}, url: {url}')
        subprocess.call(f'python scripts_faces/images_from_video.py {folder} {i} '
                        f'{data_folder_path}/videos/{folder}_{i}.mp4 '
                        f'--data_folder {data_folder_path}', shell=True)

# Cropping face from images
if images_path is not None:
    subprocess.call(f'python scripts/images_from_images.py {folder} {images_path} '
                    f'--data_folder {data_folder_path}', shell=True)

# Mask
# TODO: for images too
print('Preparing binary mask..')
subprocess.call(f'python scripts_faces/prep_binary_mask.py '
                f'--data_folder {data_folder_path}', shell=True)
