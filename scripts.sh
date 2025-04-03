python main.py --experiment_file=parameters/cross/deepglobe.yaml
python main.py --experiment_file=parameters/cross/isic.yaml
python main.py --experiment_file=parameters/cross/lung.yaml

python main.py --experiment_file=parameters/pascal/hdmnet_N1K5.yaml --multi_gpu
python main.py --experiment_file=parameters/pascal/hdmnet_N2K5.yaml --multi_gpu

python main.py --experiment_file=parameters/dcama_rank_test.yaml
python main.py --experiment_file=parameters/dcama_adapter_test.yaml
python main.py --experiment_file=parameters/dcama_encoder_test.yaml