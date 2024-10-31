import torch
from manotorch.manolayer import ManoLayer, MANOOutput
import pickle
import numpy as np
import plotly.graph_objects as go
import trimesh as tm
from scipy.spatial.transform import Rotation
from typing import Optional

dir = "data/human_hand_poses.pkl"

mano_layer = ManoLayer(use_pca=False, flat_hand_mean=True)
# mano_layer = ManoLayer(
#     rot_mode="axisang",
#     use_pca=False,
#     side="right",
#     center_idx=None,
#     mano_assets_root="assets/mano",
#     flat_hand_mean=False,
# )
def plot_mesh(mesh, color='lightpink', opacity=0.6, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)

def plot_point_cloud(pts):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode='markers',
        marker={"color":"blue","size":3},
        name="joints"
    )


file = open(dir, "rb")
data = pickle.load(file)

# times = np.array([p for p in data['pose_aa_dict'].keys()])
# print(times.shape)

# trans = data['trans']
# print(trans)
length = 17825
betas = torch.tensor(data['beta']).expand(length,10)
poses = [p[0] for p in data['pose_aa_dict'].values()]
poses = torch.tensor(np.array(poses))
poses_select = poses

pi = np.pi
rot = Rotation.from_euler("xyz",[-pi/2,pi/2,0])
rot_vec = torch.tensor([-1.2092,1.2092,1.2092])
aligned_poses = torch.zeros([1,48],dtype=torch.float32)
aligned_poses[...,:3] = rot_vec
R_1 = mano_layer(aligned_poses,betas[:1]).transforms_abs
R_1 = R_1[0][0][:3,:3].numpy()

print(poses_select.shape)
poses_select[...,:3] = 0
print(betas.shape)

# poses_select = torch.zeros_like(poses_select)
# betas = torch.zeros_like(betas)

mano_output: MANOOutput = mano_layer(poses_select,betas)

# retrieve 778 vertices, 21 joints and 16 SE3 transforms of each articulation
verts = mano_output.verts  # (B, 778, 3), root(center_joint) relative
joints = mano_output.joints  # (B, 21, 3), root relative
R_2 = mano_output.transforms_abs  # (B, 16, 4, 4), root relative

print(verts.shape) # [N, 778, 3]
print(joints.shape) # [N, 21, 3]
print(R_2.shape) # [N, 16, 4, 4]
print(R_2[0][0].shape)
R_2 = R_2[0][0][:3,:3].numpy()


R = np.linalg.inv(R_2) @ R_1

wrist_pos = joints[:,:1]
verts = verts - wrist_pos
joints = joints - wrist_pos

verts = verts.numpy()
joints = joints.numpy()
faces = mano_layer.th_faces

# for vert,joint in zip(verts,joints):
#     mesh = tm.Trimesh(vert,faces=faces)
#     fig_mesh = plot_mesh(mesh)
#     fig_point = plot_point_cloud(joint,marker={"color":"blue","size":3},name="joints")
#     go.Figure([fig_mesh,fig_point]).show()
# for i in range(3):
#     mesh_start = plot_mesh(tm.Trimesh(verts[i],faces=faces))
#     go.Figure([mesh_start]).show()
# for i in range(3):
#     mesh_start = plot_mesh(tm.Trimesh(verts[length - 1 - i],faces=faces))
#     go.Figure([mesh_start]).show()
vert_rot = verts[0]
joint_rot = joints[0]

mesh_start = plot_mesh(tm.Trimesh(vert_rot,faces=faces))
points_start = plot_point_cloud(joint_rot)
frames = []
for i in range(0,length,32):
    mesh = plot_mesh(tm.Trimesh(verts[i],faces=faces))
    points = plot_point_cloud(joints[i])
    frame = go.Frame(data=[mesh,points])
    frames.append(frame)

fig = go.Figure(data=[mesh_start,points_start], frames=frames)

fig.update_layout(
    scene=dict(
        aspectmode='cube',
        xaxis=dict(range=[-0.1, 0.1]),
        yaxis=dict(range=[-0.1, 0.1]),
        zaxis=dict(range=[0, 0.2])
    ),
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 100,
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