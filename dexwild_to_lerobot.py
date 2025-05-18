import shutil

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

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
from pose_utils import poses7d_to_mats, mats_to_poses7d
import tyro

REPO_NAME = "adakan4/dexwild_spray"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]  # For simplicity we will combine multiple Libero datasets into one training dataset

pybullet.connect(pybullet.DIRECT)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
xarmID = pybullet.loadURDF("xarm/xarm6_robot.urdf")

def main(data_dir: str, *, push_to_hub: bool = False):
    data_folder = data_dir
    action_keys = ["right_leapv1", "right_arm_eef"]
    img_keys = ["right_pinky_cam", "right_thumb_cam"]
    traj_folders = glob(f"{data_folder}/*")

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=10,
        features={
            "right_thumb_image": {
                "dtype": "image",
                "shape": (320, 240, 3),
                "names": ["height", "width", "channel"],
            },
            "right_pinky_image": {
                "dtype": "image",
                "shape": (320, 240, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (22,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (22,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    # Loop over each episode folder
    for traj_folder in tqdm(traj_folders):
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
        with open(f"{traj_folder}/right_leapv1/right_leapv1.pkl", "rb") as f:
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
                right_thumb_img = Image.open(os.path.join(os.path.expanduser("~"), "summer-work", "data_spray", f"{traj_folder}", "right_thumb_cam", f"{timestep}.jpg"))
                right_pinky_img = Image.open(os.path.join(os.path.expanduser("~"), "summer-work", "data_spray", f"{traj_folder}", "right_pinky_cam", f"{timestep}.jpg"))

                dataset.add_frame(
                        {
                            "right_thumb_image": right_thumb_img,
                            "right_pinky_image": right_pinky_img,
                            "state": state_data[idx],
                            "actions": action_data[idx],
                        }
                    )
                dataset.save_episode(task="pick up and use the spray bottle")
    
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
