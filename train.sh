var=$(pwd)
echo PWD: $var
rm -rf data/processed_data

echo "Splitting data into train and valid...."
python3 yolov5_obb/yolov5-utils/data_split.py --datapath data/oriented-object-detection-dataset/dota_rajan --outpath data/processed_data
echo "Done splitting data"
echo "Copying labes to labelTxt folder"
cd data/processed_data;ln -s labels labelTxt
cd ../..
echo "Done.."



train_data="${var}/data/processed_data/images/train/"
test_data="${var}/data/processed_data/images/valid/"
cat << EOS > yolov5_obb/data/rajan_dota.yaml

train: $train_data
val:  $train_data

# Classes
nc: 15  # number of classes
names: ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 
        'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  
        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']  #

EOS

echo Config file written at: $var/yolov5_obb/data/rajan_dota.yaml

echo Starting Training... 
pip install 'protobuf<=3.20.1' --force-reinstall
cd yolov5_obb
python3 train.py --data data/rajan_dota.yaml \
    --cfg models/yolov5m.yaml\
    --batch-size 8\
    --device 0\
    --patience 100\
    --epochs 200\
    --project $var/results/train