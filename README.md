# MobileNetv2
## WIndows
python -m venv venv
venv\Scripts\activate

##Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python prepare_dataset.py
python check_dataset.py
python train.py
python predict.py Images/img2.jpg
