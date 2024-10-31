name="svh"
python3 render_robot_hand.py \
  --pickle-path data/${name}_hand_poses.pkl \
  --robot-glob-path data/grab/hand_glob_poses.pkl \
  --object-path data/grab/object_poses.pkl \
  # --output-video-path data/grab/demo.mp4 \
  # --headless