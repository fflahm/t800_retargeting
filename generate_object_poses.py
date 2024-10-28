import math
import numpy as np
import pickle
import tyro
from scipy.spatial.transform import Rotation

def theta_to_rotation(theta):
    rot = Rotation.from_euler("xyz",[0,theta,0],degrees=False)
    return rot

# hand config
num_keypoints = 5
human_indices = [4,8,12,16,20]

# object config
half = 0.012
obj_ctr = np.array([0.05, 0.0, 0.135])
local_position = [[0,0,-half-0.004],[-half,-half,half],[0,0,half+0.006],[half-0.004,-half-0.002,half+0.008],[half,-half,-half]]

pi = np.pi

def main(position_path: str, pose_path: str,
         seq_len: int = 500, init_theta: float = -pi/6, total_theta: float = pi/3):
    
    thetas = []
    joints = np.zeros([seq_len,21,3])

    theta = init_theta
    dlt_theta = total_theta / seq_len

    for i in range(seq_len):
        theta += dlt_theta
        rot = theta_to_rotation(theta)
        for j in range(num_keypoints):
            joints[i][human_indices[j]] = obj_ctr + rot.apply(local_position[j])
        thetas.append(rot.as_quat().tolist())
    with open(position_path,"wb") as file:
        pickle.dump(joints, file)
    with open(pose_path, "wb") as file:
        pickle.dump({"center":obj_ctr,"half_len":half,"angles":thetas,"local_positions":local_position}, file)

if __name__ == "__main__":
    tyro.cli(main)