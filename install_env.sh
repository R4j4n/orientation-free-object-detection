conda create -n yolo_rotated python=3.9 -y 
conda activate yolo_rotated

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -r requirements.txt
cd yolov5_obb/utils/nms_rotated
python setup.py develop 
pip install "numpy<1.24.0"
cd ../../..   
