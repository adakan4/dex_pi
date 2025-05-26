from openpi_client import image_tools
from openpi_client import websocket_client_policy
from glob import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
import contextlib
import io
import pickle as pkl
import numpy as np
import pybullet
import pybullet_data
from scipy.spatial.transform import Rotation
from pose_utils import poses7d_to_mats, mats_to_poses7d

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)


pybullet.connect(pybullet.DIRECT)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
xarmID = pybullet.loadURDF("xarm/xarm6_robot.urdf")

data_dir = "/home/alfredo/telekinesis_3/all_data/data_01_23_robot_toy"

data_folder = data_dir
action_keys = ["right_leapv2", "right_arm_eef"]
img_keys = ["right_pinky_cam", "right_thumb_cam"]
traj_folders = glob(f"{data_folder}/*")

# Loop over each episode folder
traj_folder = traj_folders[0]
timesteps = []
file_path = f"{traj_folder}/timesteps/timesteps.txt"
try:
    with open(file_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line into individual strings
            string_numbers = line.strip().split()
            # Convert each string to an integer and append it to the timesteps list
            for num_str in string_numbers:
                try: timesteps.append(int(num_str))
                except ValueError: print(f"Invalid number found: {num_str}")     
except FileNotFoundError: print(f"Error: Timesteps file not found at '{file_path}'")
# Load the state and action data
with open(f"{traj_folder}/right_arm_eef/right_arm_eef.pkl", "rb") as f:
    arm_state_quats = pkl.load(f)[:, 1:]  # Skip the first column (timestep)
with open(f"{traj_folder}/right_leapv2/right_leapv2.pkl", "rb") as f:
    leap_state_data = pkl.load(f)[:, 1:]  # Skip the first column (timestep)

# switch to homogeneous matrices to apply the Xarm-Leap offset transformation
arm_state_mats = poses7d_to_mats(arm_state_quats)
trans_mat = np.eye(4)
trans_mat[2, 3] = -0.15
trans_mat[0, 0] = 0
trans_mat[1, 1] = 0
trans_mat[0, 1] = 1
trans_mat[1, 0] = -1 
arm_state_mats = np.array([np.matmul(arm_state_mats[i], trans_mat) for i in range(len(arm_state_mats))])
arm_state_data = mats_to_poses7d(arm_state_mats)

# calculate inverse kinematics (from XYZ and quaternion to joint angles)
arm_state_data = np.array([np.array(pybullet.calculateInverseKinematics(xarmID, 6, arm_state_data[i, 0:3], arm_state_data[i, 3:7])) for i in range(len(arm_state_data))])
if (len(arm_state_data) != len(leap_state_data) or len(arm_state_data) != len(timesteps) or len(leap_state_data) != len(timesteps)):
    print("The lengths of the data do not match! Skipping this episode.")
    print("Length of arm state data: ", len(arm_state_data))
    print("Length of leap state data: ", len(leap_state_data))
    print("Length of timesteps: ", len(timesteps))
else:
    state_data = np.append(arm_state_data, leap_state_data, axis=1)
    action_data = state_data.copy()

    # the action is simply the state at the next timestep
    for idx in range(0, len(action_data)-1):
        action_data[idx] = state_data[idx+1]


    for idx in range(min(len(action_data), len(state_data))):
        timestep = timesteps[idx]
        # Load the image data
        right_thumb_img = np.asarray(Image.open(f"{traj_folder}/right_thumb_cam/{timestep}.jpg"))
        right_pinky_img = np.asarray(Image.open(f"{traj_folder}/right_pinky_cam/{timestep}.jpg"))
        task_instruction = "clean up the toys"
        # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
        # We provide utilities for resizing images + uint8 conversion so you match the training routines.
        # The typical resize_size for pre-trained pi0 models is 224.
        # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
        
        observation = {
            "observation/right_thumb_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_thumb_img, 224, 224)
            ),
            "observation/right_pinky_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_pinky_img, 224, 224)
            ),
            "observation/state": state_data[idx],
            "prompt": task_instruction,
        }

        # Call the policy server with the current observation.
        # This returns an action chunk of shape (action_horizon, action_dim).
        # Note that you typically only need to call the policy every N steps and execute steps
        # from the predicted action chunk open-loop in the remaining steps.
        if idx > 0:
            action_chunk = client.infer(observation)["actions"]
            
            # Print the norm difference between the action chunk and the action data
            diff_norm = np.linalg.norm(action_chunk[0] - action_data[idx])
            print("Norm difference:", diff_norm)
            if diff_norm > 0.1:
                print("Warning: Action difference is large!")
            else:
                print("Predicted Action: ", action_chunk[0] - action_data[idx-1])
                print("Ground Truth Action: ", action_data[idx]- action_data[idx-1])

            diff_norm = np.linalg.norm(action_chunk[0] - action_data[idx-1])
            print("Control difference:", diff_norm)
        

        # Execute the actions in the environment.

for step in range(num_steps):
    # Inside the episode loop, construct the observation.

    ...