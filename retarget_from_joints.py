import pickle
from pathlib import Path

import cv2
import tqdm
import tyro

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


pre_freq = 1
post_freq = 1

def retarget_video(retargeting: SeqRetargeting, input_path: str, output_path: str, config_path: str):
    data = []

    with open(input_path,"rb") as file:
        joints_pos = pickle.load(file)

    joints_pos = joints_pos[::pre_freq]
    length = len(joints_pos)

    with tqdm.tqdm(total=length) as pbar:
        for joint_pos in joints_pos:

            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            data.append(qpos)
            pbar.update(1)

        data = data[::post_freq]
        meta_data = dict(
            config_path=config_path,
            dof=len(retargeting.optimizer.robot.dof_joint_names),
            joint_names=retargeting.optimizer.robot.dof_joint_names,
        )

        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(dict(data=data, meta_data=meta_data), f)

        retargeting.verbose()


def main(
    robot_name: RobotName, input_path: str, output_path: str, retargeting_type: RetargetingType, hand_type: HandType
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        input_path: The file path for the input video in .mp4 format.
        output_path: The file path for the output data in .pickle format.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
    """
    # input_path = "data/human_hand_joints.pkl"
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = "assets/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    retarget_video(retargeting, input_path, output_path, str(config_path))


if __name__ == "__main__":
    tyro.cli(main)
