source activate synth-env
export LD_PRELOAD=""
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd 
python train.py