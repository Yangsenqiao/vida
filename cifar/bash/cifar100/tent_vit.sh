export PYTHONPATH=.
conda deactivate
conda activate vida

python cifar100c_vit.py --cfg cfgs/cifar100/tent.yaml --checkpoint [your-ckpt-path] --data_dir [your-data-dir] 