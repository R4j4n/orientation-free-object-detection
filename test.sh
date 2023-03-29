var=$(pwd)
echo PWD: $var

python yolov5_obb/detect.py\
    --weights $var/results/train/exp3/weights/best.pt\
    --source $var/data/processed_data/images/valid\
    --device 0\
    --hide-conf\
    --hide-labels\
    --project $var/results/infer/ 

    