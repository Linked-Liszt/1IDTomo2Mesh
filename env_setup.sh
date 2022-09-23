# Assumes fresh conda env python=3.9
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install -r requirements_pip.txt