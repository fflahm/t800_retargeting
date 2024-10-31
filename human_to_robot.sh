name="svh"
python retarget_from_joints.py \
  --robot-name $name \
  --input-path data/grab/grab_hand_joints.pkl \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/${name}_hand_poses.pkl \
  --config-tag from_human_ \
  --post-freq 1
python3 render_robot_hand.py \
  --pickle-path data/${name}_hand_poses.pkl \
  --robot-glob-path data/grab/grab_glob_poses.pkl \
  --object-path data/grab/object_poses.pkl