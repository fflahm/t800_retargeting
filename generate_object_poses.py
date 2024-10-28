import math
import numpy as np
import pickle
from scipy.spatial.transform import Rotation

def theta_to_rotation(theta):
    rot = Rotation.from_euler("xyz",[0,theta,0],degrees=False)
    return rot

seq_len = 500
num_keypoints = 5
human_indices = [4,8,12,16,20]

half = 0.012
obj_ctr = np.array([0.05, 0.0, 0.135])

local_position = [[0,0,-half-0.004],[-half,-half,half],[0,0,half+0.006],[half-0.004,-half-0.002,half+0.008],[half,-half,-half]]

pi = np.pi
theta = - pi / 6
dlt_theta = pi / 3 / seq_len


joints = np.zeros([seq_len,21,3])
thetas = []

for i in range(seq_len):
    theta += dlt_theta
    rot = theta_to_rotation(theta)
    for j in range(num_keypoints):
        joints[i][human_indices[j]] = obj_ctr + rot.apply(local_position[j])
    thetas.append(rot.as_quat().tolist())
# print(joints[0][4])
# print(joints[0][12])

save_path = "data/object_positions.pkl"
with open(save_path,"wb") as file:
    pickle.dump(joints, file)

angles_path = "data/object_poses.pkl"
with open(angles_path, "wb") as file:
    pickle.dump({"center":obj_ctr,"half_len":half,"angles":thetas,"local_positions":local_position}, file)