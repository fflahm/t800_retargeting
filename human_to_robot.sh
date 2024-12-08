name="shadow"
filename="watch_set_1.pkl"
python3 retarget_from_joints.py \
  --robot-name $name \
  --input-path data/grab/s1/${filename} \
  --retargeting-type vector \
  --hand-type right \

python3 run_manipulation.py \
  --pickle-path data/grab/s1/${filename} \
  # --output-video-path data/demo.mp4 \
  # --headless