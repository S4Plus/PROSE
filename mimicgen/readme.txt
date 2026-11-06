conda activate mimicgen
#项目根目录
cd mimicgen

#下载数据集   core中已有1000条数据
python mimicgen/scripts/download_datasets.py

#转化成lerobotv2.1格式
python myscripts/convert_hdf5_to_lerobot.py  --input ./datasets/core/square_d1.hdf5 --output-dir ./datasets/lerobot/lerobot_square_d1/ --max-episodes 0

#训练