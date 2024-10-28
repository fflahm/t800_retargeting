from pathlib import Path
from typing import Optional, List, Union, Dict
import math
import cv2
import numpy as np
import sapien
import tqdm
import tyro
import pickle
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer


from dex_retargeting.retargeting_config import RetargetingConfig
from scipy.spatial.transform import Rotation

# Convert webp
# ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp

def theta_to_quaternion(theta):
    return [math.cos(theta/2),math.sin(theta/2),0.0,0.0]

def euler_to_quaternion(rpy):
    return Rotation.from_euler("xyz",rpy,degrees=False).as_quat()

np.set_printoptions(precision=3)
pi = np.pi

def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):  

    # Generate rendering config
    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    # Config is loaded only to find the urdf path and robot name
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()

    # Ground
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)

    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.12, 0.12, 0.15], euler_to_quaternion([-pi/4,0.0,0.0])))


    # Viewer
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None
    record_video = output_video_path is not None

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    # if "ability" in robot_name:
    #     loader.scale = 1.5
    # elif "dclaw" in robot_name:
    #     loader.scale = 1.25
    # elif "allegro" in robot_name:
    #     loader.scale = 1.4
    # elif "shadow" in robot_name:
    #     loader.scale = 0.9
    # elif "bhand" in robot_name:
    #     loader.scale = 1.5
    # elif "leap" in robot_name:
    #     loader.scale = 1.4
    # elif "svh" in robot_name:
    #     loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    # if "ability" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "shadow" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.2]))
    # elif "dclaw" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "allegro" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.05]))
    # elif "bhand" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.2]))
    # elif "leap" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))
    # elif "svh" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.13]))
    # elif "inspire" in robot_name:
    #     robot.set_pose(sapien.Pose([0, 0, -0.15]))

    # Create object
    obj_path = "data/object_poses.pkl"
    with open(obj_path,"rb") as file:
        obj_data = pickle.load(file)
        initial_pos = obj_data["center"]
        half_len = obj_data["half_len"]
        thetas = obj_data["angles"]
        local_positions = obj_data["local_positions"]


    if (len(data) != len(thetas)):
        raise ValueError("Hand and object length not match!")
    initial_theta = 0.0
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_collision(half_size=[half_len,half_len,half_len])
    actor_builder.add_box_visual(half_size=[half_len,half_len,half_len], material=[1.0, 0.0, 0.0])
    box = actor_builder.build(name="box")  # Add a box
    box.set_pose(sapien.Pose(p=initial_pos,q=theta_to_quaternion(initial_theta)))

    # Video recorder
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (cam.get_width(), cam.get_height())
        )

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array([retargeting_joint_names.index(name) for name in sapien_joint_names]).astype(int)
    
    key_link_names = ["thtip", "fftip", "mftip", "rftip", "lftip"]
        
    # data = np.zeros([2500,20])
    # for i in range(20):
    #     data[100*(i+1):100*(i+2)][:,i] = 1.0
    # thetas = np.zeros([2500,4]) 

    for qpos, theta in tqdm.tqdm(zip(data,thetas)):

        
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])
        box.set_pose(sapien.Pose(p=initial_pos,q=theta))
        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])


    # Show acc
    last_rot = Rotation.from_quat(theta)
    for link_name, local_pos in zip(key_link_names,local_positions):
        target_pos = robot.find_link_by_name(link_name).get_entity_pose().get_p()
        link_pos = last_rot.apply(local_pos) + initial_pos
        print(f"{link_name}: pos {link_pos} target {target_pos} rel_dist {np.linalg.norm(target_pos-link_pos)/half_len:.2f}")

    while not viewer.closed:
        
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    output_video_path: Optional[str] = None,
    headless: bool = False,
):
    """
    Loads the preserved robot pose data and renders it either on screen or as an mp4 video.

    Args:
        pickle_path: Path to the .pickle file, created by `detect_from_video.py`.
        output_video_path: Path where the output video in .mp4 format would be saved.
            By default, it is set to None, implying no video will be saved.
        headless: Set to visualize the rendering on the screen by opening the viewer window.
    """
    robot_dir = "assets/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    headless = False
    output_video_path = None
    render_by_sapien(meta_data, data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
