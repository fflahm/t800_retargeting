import torch
import numpy as np
import pickle
import trimesh as tm
import plotly.graph_objects as go
from manotorch.manolayer import ManoLayer

data_dir = "data/human_hand_poses.pkl"
save_dir = "data/mesh_seq_data.pkl"

def load_data(dir=data_dir, rot=[-1.2092,1.2092,1.2092]):
    with open(dir, "rb") as file:
        data = pickle.load(file)
    times = np.array([t for t in data['pose_aa_dict'].keys()])
    poses = [p[0] for p in data['pose_aa_dict'].values()]
    poses = torch.tensor(np.array(poses))
    poses[...,:3] = torch.tensor(rot)
    num_frames = len(poses)
    betas = torch.tensor(data['beta']).expand(num_frames,10)
    return num_frames,poses,betas,times

def get_verts_and_faces(poses,betas):
    mano_layer = ManoLayer(use_pca=False, flat_hand_mean=True)
    output = mano_layer(poses,betas)
    verts = output.verts
    joints = output.joints
    wrist_pos = joints[:,:1]
    verts = (verts - wrist_pos).numpy()
    faces = mano_layer.th_faces.numpy()
    return verts,faces

def save_data(verts, faces, times, dir=save_dir):
    with open(dir,"wb") as file:
        pickle.dump({"verts_seq":verts.tolist(),"faces":faces.tolist(),"time_seq":times.tolist()},file)

if __name__ == "__main__":
    num_frames,poses,betas,time_seq = load_data()
    verts,faces = get_verts_and_faces(poses,betas)
    save_data(verts,faces,time_seq)
