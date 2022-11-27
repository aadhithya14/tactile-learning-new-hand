import os 
import cv2
import h5py 
import numpy as np
import pickle 
from tqdm import tqdm

from holobot.robot.allegro.allegro_kdl import AllegroKDL

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

def dump_data_indices(demo_id, root):
    # Matches the index -> demo_id, datapoint_id according to the timestamps saved
    allegro_indices, image_indices, tactile_indices = [], [], []
    # for root in roots:
    allegro_states_path = os.path.join(root, 'allegro_joint_states.h5')
    image_metadata_path = os.path.join(root, 'cam_0_rgb_video.metadata')
    tactile_info_path = os.path.join(root, 'touch_sensor_values.h5')
    with h5py.File(allegro_states_path, 'r') as f:
        allegro_timestamps = f['timestamps'][()]
        allegro_positions = f['positions'][()]
    with h5py.File(tactile_info_path, 'r') as f:
        tactile_timestamps = f['timestamps'][()]
    with open(image_metadata_path, 'rb') as f:
        image_metadata = pickle.load(f)
        image_timestamps = np.asarray(image_metadata['timestamps']) / 1000.
    
    # Start the allegro kdl solver 
    allegro_kdl_solver = AllegroKDL()

    allegro_id, image_id, tactile_id = 0, 0, 0
    # Get the first timestamps
    tactile_timestamp = tactile_timestamps[0]
    allegro_id = get_closest_id(allegro_id, tactile_timestamp, allegro_timestamps)
    image_id = get_closest_id(image_id, tactile_timestamp, image_timestamps)

    tactile_indices.append([demo_id, tactile_id])
    allegro_indices.append([demo_id, allegro_id])
    image_indices.append([demo_id, image_id])

    while (True):
        # Get the proper next allegro id
        allegro_id = find_next_allegro_id(allegro_kdl_solver, allegro_positions, allegro_id)
        if allegro_id >= len(allegro_positions)-1:
            break

        # Get the closest timestamps to that
        allegro_timestamp = allegro_timestamps[allegro_id]
        tactile_id = get_closest_id(tactile_id, allegro_timestamp, tactile_timestamps)
        image_id = get_closest_id(image_id, allegro_timestamp, image_timestamps)
        print('allegro_timestamps[{}/{}]: {}, tactile_timstamps[{}/{}]: {}, image_timestamps[{}/{}]: {}'.format(
            allegro_id, len(allegro_timestamps), allegro_timestamp,
            tactile_id, len(tactile_timestamps), tactile_timestamps[tactile_id],
            image_id, len(image_timestamps), image_timestamps[image_id]
        ))

        tactile_indices.append([demo_id, tactile_id])
        allegro_indices.append([demo_id, allegro_id])
        image_indices.append([demo_id, image_id])

        # If some of the data has ended then retur it
        if image_id >= len(image_timestamps)-1 or \
           tactile_id >= len(tactile_timestamps)-1 or \
           allegro_id >= len(allegro_timestamps)-1:
            break

    # Save the indices for that root 
    with open(os.path.join(root, 'tactile_indices.pkl'), 'wb') as f:
        pickle.dump(tactile_indices, f)
    with open(os.path.join(root, 'allegro_indices.pkl'), 'wb') as f:
        pickle.dump(allegro_indices, f)
    with open(os.path.join(root, 'image_indices.pkl'), 'wb') as f:
        pickle.dump(image_indices, f)

    
def find_next_allegro_id(kdl_solver, positions, pos_id):
    old_allegro_pos = positions[pos_id]
    old_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(old_allegro_pos)
    for i in range(pos_id, len(positions)):
        curr_allegro_fingertip_pos = kdl_solver.get_fingertip_coords(positions[i])
        step_size = np.linalg.norm(old_allegro_fingertip_pos - curr_allegro_fingertip_pos)
        if step_size > 0.02: 
            return i

    return i # This will return len(positions)-1 when there are not enough 

def get_closest_id(curr_id, desired_timestamp, all_timestamps):
    for i in range(curr_id, len(all_timestamps)):
        if all_timestamps[i] > desired_timestamp:
            return i # Find the first timestamp that is after that
    
    return i

if __name__ == '__main__':
    data_dir = '/home/irmak/Workspace/Holo-Bot/extracted_data/demonstration_18'
    # dump_video_to_images(root=data_dir)
    