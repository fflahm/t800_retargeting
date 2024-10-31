import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import pickle
import numpy as np
import plotly.graph_objects as go
import trimesh as tm
from utils import plot_mesh, plot_point_cloud, get_joints_point_cloud, plotly_lines, show_plotly


mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False)


robot_path = "data/robot_hand_joints/1.pkl"
human_path = "data/mesh_seq_data.pkl"
human_joints_path = "data/not_flat_human_hand_joints.pkl"
with open(robot_path, "rb") as file:
    data = pickle.load(file)
    robot_joints:np.array =  data["robot_joints"]

    robot_joints[..., 0] /= data["scaling_factor"][0]
    robot_joints[..., 1] /= data["scaling_factor"][1]
    robot_joints[..., 2] /= data["scaling_factor"][2]

    origin_indices = data["link_indices"][0]
    target_indices = data["link_indices"][1]


with open(human_path, "rb") as file:
    data = pickle.load(file)
    human_verts = data["verts_seq"]
with open(human_joints_path, "rb") as file:
    human_joints:np.array = pickle.load(file)[::32]


human_origin_joints = human_joints[:,origin_indices,:] # [num_frames, num_tsvs, 3]
human_target_joints = human_joints[:,target_indices,:]
robot_origin_joints = human_joints[:,origin_indices,:]
robot_target_joints = robot_joints[:,target_indices,:] - robot_joints[:,origin_indices,:] + human_joints[:,origin_indices,:]


faces = mano_layer.th_faces.numpy()

# Compute losses
human_tips = human_joints[:,[4,8,12,16,20],:] 
robot_tips = robot_joints[:,[4,8,12,16,20],:]
tip_dist = np.linalg.norm(human_tips-robot_tips,axis=2)
print(f"Per fingertip position distance: {np.mean(tip_dist)}")

joint_dist = np.linalg.norm(human_joints-robot_joints, axis=2)
print(f"Per joint position distance: {np.mean(joint_dist)}")

human_tsvs = human_target_joints - human_origin_joints
robot_tsvs = robot_target_joints - robot_origin_joints
delta_tsvs = robot_tsvs - human_tsvs
tsv_dist = np.linalg.norm(delta_tsvs,axis=2)
print(f"Per TSV distance: {np.mean(tsv_dist)}")

# Figure 0: Definition of TSVs
human_hand_mesh = plot_mesh(tm.Trimesh(human_verts[0],faces),name="human_hand")
human_tsvs = plotly_lines(human_origin_joints[0],human_target_joints[0], color="blue", name="human_tsv")
human_tsvs.append(human_hand_mesh)

show_plotly(data=human_tsvs)

# Figure 1: Difference between TSVs
frames = []
start_lines = plotly_lines(human_origin_joints[0],human_target_joints[0], color="blue", name="human_tsv")
# start_lines += plotly_lines(robot_origin_joints[0],robot_target_joints[0],color="red",name="robot_tsv")
start_lines += plotly_lines(human_target_joints[0],human_target_joints[0]+delta_tsvs[0],color="yellow", name="delta_tsv")
for human_origin, human_target, robot_origin, robot_target, delta_tsv in zip(
    human_origin_joints,human_target_joints,robot_origin_joints,robot_target_joints,delta_tsvs):

    lines = plotly_lines(human_origin,human_target, color="blue", name="human_tsv")
    # lines += plotly_lines(robot_origin,robot_target,color="red",name="robot_tsv")
    lines += plotly_lines(human_target,human_target+delta_tsv,color="yellow", name="delta_tsv")
    frames.append(go.Frame(data=lines))

show_plotly(start_lines,frames)

# Figure 2: Difference between hands

points_start = get_joints_point_cloud(robot_joints[0])
points_start.append(plot_mesh(tm.Trimesh(human_verts[0],faces)))
frames = []
for joint,vert in zip(robot_joints,human_verts):
    points = get_joints_point_cloud(joint)
    points.append(plot_mesh(tm.Trimesh(vert,faces)))
    frame = go.Frame(data=points)
    frames.append(frame)

show_plotly(points_start,frames)

