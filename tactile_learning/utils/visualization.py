import cv2
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

from holobot.utils.network import ZMQCameraSubscriber

def plot_tactile_sensor(ax, sensor_values, use_img=False, img=None, title='Tip Position'):
    # sensor_values: (16, 3) - 3 values for each tactile - x and y represents the position, z represents the pressure on the tactile point
    img_shape = (240, 240, 3) # For one sensor
    blank_image = np.ones(img_shape, np.uint8) * 255
    if use_img == False: 
        img = ax.imshow(blank_image.copy())
    ax.set_title(title)

    # Set the coordinates for each circle
    tactile_coordinates = []
    for j in range(48, 192+1, 48): # Y
        for i in range(48, 192+1, 48): # X - It goes from top left to bottom right row first 
            tactile_coordinates.append([i,j])

    # Plot the circles 
    for i in range(sensor_values.shape[0]):
        center_coordinates = (
            tactile_coordinates[i][0] + int(sensor_values[i,0]/20), # NOTE: Change this
            tactile_coordinates[i][1] + int(sensor_values[i,1]/20)
        )
        radius = max(10 + int(sensor_values[i,2]/10), 2)
      
        if i == 0:
            frame_axis = cv2.circle(blank_image.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)
        else:
            frame_axis = cv2.circle(frame_axis.copy(), center_coordinates, radius, color=(0,255,0), thickness=-1)

    img.set_array(frame_axis)

    return img, frame_axis

def plot_fingertip_position(ax, tip_position, finger_index): 
    # Tip position: (3,) - (x,y,z) positions of the tip
    # finger_index: 0 or 1
    types = ['X', 'Y', 'Z']
    values = tip_position 
 
    ax.set_ylim(-0.05, 0.15)
    if finger_index == 0: # The index finger 
        ax.bar(types, values, color='darkolivegreen')
        ax.set_title('Index Finger Tip Position')
    elif finger_index == 1:
        ax.bar(types, values, color='mediumturquoise')
        ax.set_title('Middle Finger Tip Position')

def dump_tactile_state(tactile_value, allegro_tip_pos, title='Nearest Neighbor'): # Or Current State
    # tactile_value: (2,16,3)
    # allegro_tip_pos: (6,)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    plot_tactile_sensor(axs[0,0], tactile_value[0,:,:], title='Index Tip Tactile Sensors')
    plot_tactile_sensor(axs[0,1], tactile_value[1,:,:], title='Middle Tip Tactile Sensors')
    plot_fingertip_position(axs[1,0], allegro_tip_pos[0:3], 0)
    plot_fingertip_position(axs[1,1], allegro_tip_pos[3:], 1)
    fig.suptitle(title)
    fig.savefig(f'{title}.png') # And we will imshow them during deployment
    fig.clf()

def dump_camera_image(host='172.24.71.240', image_stream_port=10005):
    image_subscriber = ZMQCameraSubscriber(
        host = host,
        port = image_stream_port,
        topic_type = 'RGB'
    )
    image, _ = image_subscriber.recv_rgb_image()
    cv2.imwrite('Camera Image.png', image)

def dump_knn_state(dump_dir, img_name):
    os.makedirs(dump_dir, exist_ok=True)
    curr_state = cv2.imread('Current State.png')
    knn_state = cv2.imread('Nearest Neighbor.png')
    camera_img = cv2.imread('Camera Image.png')

    state_img = cv2.hconcat([curr_state, knn_state])
    width_scale = camera_img.shape[1] / state_img.shape[1]
    state_img = cv2.resize(
        state_img, 
        (int(state_img.shape[1] * width_scale),
         int(state_img.shape[0] * width_scale))
    )

    all_state_img = cv2.vconcat([camera_img, state_img])
    cv2.imwrite(os.path.join(dump_dir, img_name), all_state_img)
    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(all_state_img, cv2.COLOR_BGR2RGB))

    # plt.pause(0.01)
    # plt.cla()


# Example
if __name__ == '__main__':
    curr_tactile_value = np.random.rand(15,16,3)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    plot_tactile_sensor(axs[0], curr_tactile_value[3,:,:])
    plot_tactile_sensor(axs[1], curr_tactile_value[7,:,:])
    plt.show()