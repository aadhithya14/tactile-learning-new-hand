import os 
import cv2
from tqdm import tqdm

# Method to dump a video to images - in order to receive images for timestamps

# Dumping video to images
# Creating pickle files to pick images
def dump_video_to_images(root: str, video_type: str ='rgb', view_num: int=0) -> None:
    # Convert the video into image sequences and name them with the frames
    video_path = os.path.join(root, f'cam_{view_num}_{video_type}_video.avi')
    # video_path = os.path.join(root, f'videos/{video_type}_video.mp4') # TODO: this will be taken from cfg.data_dir
    images_path = os.path.join(root, f'cam_{view_num}_{video_type}_images')
    if os.path.exists(images_path):
        print(f'{images_path} exists dump_video_to_images exiting')
        return
    os.makedirs(images_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    print(f'dumping video in {root}')
    pbar = tqdm(total = frame_count)
    while success: # The matching 
        pbar.update(1)
        cv2.imwrite('{}.png'.format(os.path.join(images_path, 'frame_{}'.format(str(frame_id).zfill(5)))), image)
        success, image = vidcap.read()
        frame_id += 1

    print(f'dumping finished in {root}')

if __name__ == '__main__':
    data_dir = '/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_18'
    dump_video_to_images(root=data_dir)