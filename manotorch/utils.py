import numpy as np
import torch
import plotly.graph_objects as go
from typing import Optional
from manotorch.manolayer import ManoLayer, MANOOutput

def plot_mesh(mesh, color='lightpink', opacity=0.6, name='mesh'):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True)

def plot_point_cloud(pts, color="blue", name="points"):
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

def plotly_line(start, end, color="blue", width=6, name="line"):
    return go.Scatter3d(x=[start[0], end[0]],
                        y=[start[1], end[1]],
                        z=[start[2], end[2]],
                        mode="lines",
                        line={"color":color, "width":width},
                        name=name)

def plotly_lines(starts, ends, color="blue", width=6, name="line"):
    figs = []

    for start,end in zip(starts,ends):
        figs.append(plotly_line(start,end,color,width,name))
    return figs

def show_plotly(data, frames= None, scene_range= [[-0.1,0.1],[-0.1,0.1],[0.0,0.2]]):
    if frames == None:
        fig = go.Figure(data)
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=scene_range[0]),
                yaxis=dict(range=scene_range[1]),
                zaxis=dict(range=scene_range[2])
            )
        )
        fig.show()
    else:
        fig = go.Figure(data=data,frames=frames)
        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                xaxis=dict(range=scene_range[0]),
                yaxis=dict(range=scene_range[1]),
                zaxis=dict(range=scene_range[2])
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
                    ]
                )
            ]
        )
        fig.show()

def generate_mano_outputs(mano_layer:ManoLayer, poses:np.ndarray, beta:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    poses = torch.tensor(poses, dtype=torch.float32) # [N, 48]
    seq_len = len(poses)
    betas = torch.tensor(beta, dtype=torch.float32).expand([seq_len, 10]) # [N, 10]
    output:MANOOutput = mano_layer(poses, betas)
    verts_seq = output.verts.numpy()
    joints_seq = output.joints.numpy()
    return verts_seq, joints_seq
