import sapien

scene = sapien.Scene()
# A small timestep for higher control accuracy
scene.set_timestep(1 / 2000.0)
scene.add_ground(0)

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

viewer = scene.create_viewer()
viewer.set_camera_xyz(x=-2, y=0, z=1)
viewer.set_camera_rpy(r=0, p=-0.3, y=0)

# Load URDF
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load("jaco2/jaco2.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

# Set joint positions
arm_zero_qpos = [4.71, 2.84, 0.0, 0.75, 4.62, 4.48, 4.88]
gripper_init_qpos = [0, 0, 0, 0, 0, 0]
zero_qpos = arm_zero_qpos + gripper_init_qpos
robot.set_qpos(zero_qpos)
arm_target_qpos = [4.71, 2.84, 0.0, 0.75, 4.62, 4.48, 4.88]
target_qpos = arm_target_qpos + gripper_init_qpos

active_joints = robot.get_active_joints()
for joint_idx, joint in enumerate(active_joints):
    joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")
    joint.set_drive_target(target_qpos[joint_idx])
    # Or you can directly set joint targets for an articulation
    # robot.set_drive_target(target_qpos)

while not viewer.closed:
    for _ in range(4):  # render every 4 steps
        # qf = robot.compute_passive_force(
        #     gravity=True,
        #     coriolis_and_centrifugal=True,
        # )
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")
            joint.set_drive_target(target_qpos[joint_idx])
        # robot.set_qf(qf)
        scene.step()
    scene.update_render()
    viewer.render()