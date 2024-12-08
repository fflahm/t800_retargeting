name="shadow"
python3 retarget_from_joints.py \
  --robot-name $name \
  --input-path data/grab/s1/cup_lift.pkl \
  --retargeting-type vector \
  --hand-type right \
  # --output-path data/svh_hand_poses.pkl \
  # --config-tag from_object_
