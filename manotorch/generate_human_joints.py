import torch
import numpy as np
import pickle
from manotorch.manolayer import ManoLayer
from scipy.spatial.transform import Rotation

data_dir = "data/human_hand_poses.pkl"
grab_data_dir = "data/grab/apple_lift.npz"
grab_beta_dir = "data/grab/s1_rhand_betas.npy"
save_dir = "data/grab/grab_hand_joints.pkl"

def load_data(dir=data_dir, rot=[-1.2092,1.2092,1.2092]):
    with open(dir, "rb") as file:
        data = pickle.load(file)
    poses = [p[0] for p in data['pose_aa_dict'].values()]
    poses = torch.tensor(np.array(poses))
    poses[...,:3] = torch.tensor(rot)
    num_frames = len(poses)
    betas = torch.tensor(data['beta']).expand(num_frames,10)
    return num_frames,poses,betas

def load_grab(data_path = grab_data_dir, beta_path = grab_beta_dir, rot=[-1.2092,1.2092,1.2092]):
    data_dict = np.load(data_path, allow_pickle=True)
    rhand = data_dict["rhand"].item()['params']
    rot_seq = rhand['global_orient'] # [N, 3] rotation vector
    tran_seq = rhand['transl'] # [N, 3]
    pose_seq = rhand['fullpose'] # [N, 45]
    beta = np.load(beta_path) # [10]

    pose_seq = np.concatenate((rot_seq, pose_seq), axis=1)
    poses = torch.tensor(pose_seq, dtype=torch.float32)[200:300]
    poses[...,:3] = torch.tensor(rot)
    num_frames = len(poses)
    betas = torch.tensor(beta, dtype=torch.float32).expand(num_frames, 10)

    # glob_poses = dict()
    # glob_poses["tran_seq"] = tran_seq[200:300]
    # glob_poses["rot_seq"] = Rotation.from_rotvec(rot_seq).as_quat()[200:300]
    # with open("data/grab/grab_glob_poses.pkl","wb") as file:
    #     pickle.dump(glob_poses,file)

    return num_frames, poses, betas


def get_joints_pos(poses,betas):
    mano_layer = ManoLayer(use_pca=False, flat_hand_mean=True)
    output = mano_layer(poses,betas)
    joints = output.joints
    wrist_pos = joints[:,:1]
    joints = joints - wrist_pos
    return joints

def save_joints_pos(joints, dir=save_dir):
    with open(dir,"wb") as file:
        pickle.dump(joints,file)

if __name__ == "__main__":
    # num_frames,poses,betas = load_data()
    num_frames,poses,betas = load_grab()
    joints = get_joints_pos(poses,betas)
    joints = joints.numpy()
    save_joints_pos(joints)
