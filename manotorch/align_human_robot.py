import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import pickle
import numpy as np
import plotly.graph_objects as go
import trimesh as tm
from utils import plot_mesh, plot_point_cloud, get_joints_point_cloud, plotly_lines, show_plotly

from torch.optim.adam import Adam
from tqdm import trange, tqdm

human_path = "data/not_flat_human_hand_joints.pkl"
robot_path = "data/svh_hand_joints.pkl"

device = "cpu"
lr = 0.01
num_steps = 1000

with open(human_path, "rb") as file:
    human_joints = pickle.load(file)[0] # [21, 3]
with open(robot_path, "rb") as file:
    robot_joints = pickle.load(file)[0] # [21, 3]


# human_joints = human_joints[[0,4,8,12,16,20]]
# robot_joints = robot_joints[[0,4,8,12,16,20]]


human_joints = torch.tensor(human_joints, device=device)
robot_joints = torch.tensor(robot_joints, device=device)
origins = torch.zeros_like(human_joints)

lines = plotly_lines(origins, human_joints, color="blue", name="human")
lines += plotly_lines(origins, robot_joints, color="red", name="robot")
show_plotly(lines)

argmin_scale = None
min_dist = 100

for x_scale in np.arange(0.5,2.0,0.1):
    for y_scale in np.arange(0.8,2.0,0.1):
        for z_scale in np.arange(0.8,2.0,0.1):
            scaled = human_joints.clone()
            scaled[:,0] *= x_scale
            scaled[:,1] *= y_scale
            scaled[:,2] *= z_scale
            dist = torch.norm(scaled - robot_joints, dim=1).mean()
            if dist < min_dist:
                min_dist = dist
                argmin_scale = [x_scale,y_scale,z_scale]
print(min_dist)
print(argmin_scale)

scaled = human_joints.clone()
scaled[:,0] *= argmin_scale[0]
scaled[:,1] *= argmin_scale[1]
scaled[:,2] *= argmin_scale[2]
lines = plotly_lines(origins, scaled, color="blue", name="human")
lines += plotly_lines(origins, robot_joints, color="red", name="robot")
show_plotly(lines)

exit()

scale = torch.tensor(1.0, device=device, requires_grad= False)




optimizer = Adam([{"params": scale, "lr": lr}])

for i in trange(num_steps):
    scaled_human_joints = human_joints * scale
    loss = torch.norm(scaled_human_joints - robot_joints, dim=1).sum()
    # loss = torch.sum((scaled_human_joints - robot_joints)**2)
    loss.backward()
    optimizer.step()
    tqdm.write(f"Step {i}: Loss {loss.detach().item()}")

print(scale)