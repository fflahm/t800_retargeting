name="svh"
python3 retarget_from_joints.py \
  --robot-name $name \
  --input-path data/object_positions.pkl \
  --retargeting-type vector \
  --hand-type right \
  --output-path data/svh_hand_poses.pkl