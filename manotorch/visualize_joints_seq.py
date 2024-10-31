import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import pickle
import numpy as np
import plotly.graph_objects as go
import trimesh as tm



mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False)
 
def plot_mesh(mesh, color='lightpink', opacity=0.6, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)

def plot_point_cloud(pts, color="blue", name="joints"):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={"color":color,"size":3},
        name=name
    )

def get_joints_point_cloud(joint): # [21, 3]

    wrist = plot_point_cloud(joint[:1],"black","wrist")
    mcp = plot_point_cloud(joint[[1,5,9,13,17]],"red","mcp")
    pip = plot_point_cloud(joint[[2,6,10,14,18]],"blue","pip")
    dip = plot_point_cloud(joint[[3,7,11,15,19]],"green","dip")
    tip = plot_point_cloud(joint[[4,8,12,16,20]],"yellow","tip")
    return [wrist,mcp,pip,dip,tip]

def get_tips_point_cloud(joint, color="black"): # [21, 3]

    
    return [plot_point_cloud(joint[[0,4,8,12,16,20]], color,"tips")]

# path = "data/example_hand_joints.pkl"
path = "data/grab/grab_hand_joints.pkl"
with open(path, "rb") as file:
    joints_seq = pickle.load(file) # [N, 21, 3]

scaling = 1
# robot_path = "data/svh_example_hand_joints.pkl"
robot_path = "data/grab/grab_hand_joints.pkl"
with open(robot_path, "rb") as file:
    robot_joints_seq = pickle.load(file) / scaling # [N, 21, 3]
print(robot_joints_seq.shape)




points_start = get_joints_point_cloud(joints_seq[0])
points_start += get_tips_point_cloud(robot_joints_seq[0])
frames = []
for joint,r_joint in zip(joints_seq,robot_joints_seq):
    points = get_joints_point_cloud(joint)
    points += get_tips_point_cloud(r_joint)
    frame = go.Frame(data=points)
    frames.append(frame)

fig = go.Figure(data=points_start, frames=frames)

fig.update_layout(
    scene=dict(
        aspectmode='cube',
        xaxis=dict(range=[-0.1, 0.1]),
        yaxis=dict(range=[-0.1, 0.1]),
        zaxis=dict(range=[0,0.2])
    ),
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 20, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 20,
                                                                      "easing": "quadratic-in-out"}}]),
                # dict(label="Pause",
                #      method="animate",
                #      args=[None, {"frame": {"duration": 0, "redraw": False},
                #                   "mode": "immediate",
                #                   "transition": {"duration": 0}}])
            ]
        )
    ]
)

fig.show()