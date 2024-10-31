import torch
import numpy as np
import pickle
import trimesh as tm
import plotly.graph_objects as go
from manotorch.manolayer import ManoLayer
from scipy.spatial.transform import Rotation
from typing import Optional


data_dir = "data/human_hand_poses.pkl"
mano_layer = ManoLayer(use_pca=False, flat_hand_mean=True)
align_rotvec = [-1.2092,1.2092,1.2092]

def load_data(dir=data_dir):
    with open(dir, "rb") as file:
        data = pickle.load(file)
    times = np.array([t for t in data['pose_aa_dict'].keys()])
    poses = [p[0] for p in data['pose_aa_dict'].values()]
    poses = torch.tensor(np.array(poses))
    poses = poses[::32]
    num_frames = len(poses)
    betas = torch.tensor(data['beta']).expand(num_frames,10)
    print(f"{num_frames} frames collected.")
    return num_frames,poses,betas,times

def get_verts(poses, betas, target_rot=Rotation.from_rotvec(align_rotvec), poses_mask:Optional[torch.tensor] = None):
    if poses_mask is not None:
        poses[...,:3] = poses_mask
    output = mano_layer(poses,betas)
    verts = output.verts
    joints = output.joints
    wrist_pos = joints[:,:1]
    verts = (verts - wrist_pos).numpy()

    current_rots = poses.numpy()[...,:3] # [N, 3]

    for i in range(len(poses)):
        current_rot = Rotation.from_rotvec(current_rots[i])
        verts[i] = target_rot.apply(current_rot.inv().apply(verts[i]))
        
    
    return verts

def plot_mesh(mesh, color='lightpink', opacity=0.6, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)


def visualize_mesh_seq(verts, freq=32, duration=20):
    faces = mano_layer.th_faces.numpy()
    mesh_start = plot_mesh(tm.Trimesh(verts[0],faces=faces))
    frames = []
    for i in range(0,len(verts),freq):
        mesh = plot_mesh(tm.Trimesh(verts[i],faces=faces))
        frame = go.Frame(data=[mesh])
        frames.append(frame)
    fig = go.Figure(data=[mesh_start], frames=frames)
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
                        args=[None, {"frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": duration,
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

def visualize_meshes_seq(verts_list, name_list, color_list, offset_list, freq=32, duration=20):
    faces = mano_layer.th_faces.numpy()
    meshes_start = []
    
    num_meshes = len(name_list)
    verts_list = np.array(verts_list) # [M, N, 778, 3]
    offset_list = np.array(offset_list) # [M,]
    trans_list = np.zeros([num_meshes,3]) # [M, 3]
    trans_list[:,1] = offset_list
    trans_list = trans_list[:,np.newaxis,np.newaxis,:] # [M, 1, 1, 3]
    verts_list = verts_list + trans_list

    for verts,name,color in zip(verts_list,name_list,color_list):
        mesh_start = plot_mesh(tm.Trimesh(verts[0],faces=faces),name=name)
        meshes_start.append(mesh_start)
    frames = []
    for i in range(0,len(verts_list[0]),freq):
        meshes = []
        for verts,name,color in zip(verts_list,name_list,color_list):
            mesh = plot_mesh(tm.Trimesh(verts[i],faces=faces),name=name,color=color)
            meshes.append(mesh)
        frame = go.Frame(data=meshes)
        frames.append(frame)
    print(f"{len(frames)} frames visualized.")
    fig = go.Figure(data=meshes_start, frames=frames)
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[-0.2, 0.2]),
            yaxis=dict(range=[-0.2, 0.2]),
            zaxis=dict(range=[-0.2, 0.2])
        ),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": duration,
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

if __name__ == "__main__":
    num_frames,poses,betas,time_seq = load_data()
    verts = get_verts(poses,betas)
    verts_align = get_verts(poses,betas,poses_mask=torch.tensor(align_rotvec))
    visualize_meshes_seq([verts,verts_align],['origin_rot','align_rot'],['lightpink','lightblue'],[0,0],freq=1)
