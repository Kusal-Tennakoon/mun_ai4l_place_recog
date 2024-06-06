#!/bin/sh
sudo apt-get update -y
sudo apt-get install python3 -y
python3 --version
sudo apt-get install python3-pip -y
pip3 --version
sudo apt-get install python3-scipy -y
pip install -U scikit-learn
pip install faiss-cpu
pip install tensorflow
pip install opencv-contrib-python
pip install seaborn
sudo apt install pypy -y
# sudo snap install pypy --classic