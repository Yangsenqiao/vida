export PYTHONPATH=.
conda deactivate
conda activate vida

python cifar10c_vit.py --cfg cfgs/cifar10/source.yaml --checkpoint [your-ckpt-path] --data_dir [your-data-dir] 