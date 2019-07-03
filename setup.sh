# conda create -n tobaco python=3.7 -y
# source activate tobaco
conda install cython -y
conda install pytorch torchvision cudatoolkit=8.0 -c pytorch -y
pip install opencv-python
pip install numpy
cd mmdetection
./compile.sh
python setup.py develop
