dir="data/grab/s1"
for filename in "$dir"/*; do
    echo "Retargeting $filename"
    python3 retarget_from_joints.py \
    --robot-name shadow \
    --input-path $filename \
    --retargeting-type vector \
    --hand-type right 
done