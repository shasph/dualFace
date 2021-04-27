call conda remove -n py36df
call conda create -n py36df python=3.6 
call conda activate py36df
call conda install pytorch==1.3.1 -c pytorch
pip install cmake
pip install -r requirements.txt
pause