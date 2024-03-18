export PYTHONPATH=
conda deactivate
conda activate vida

python imagenetc.py --cfg ./cfgs/vit/vida.yaml --checkpoint [your-ckpt-path] --data_dir [your-data-dir] 
