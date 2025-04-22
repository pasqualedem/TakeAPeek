# CD-FSS
python main.py --experiment_file=parameters/cross/deepglobe --sequential
python main.py --experiment_file=parameters/cross/isic.yaml --sequential
python main.py --experiment_file=parameters/cross/lung.yaml --sequential

# FSS
python main.py --experiment_file=parameters/pascal/hdmnet_N1K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/hdmnet_N2K5.yaml --sequential
python main.py --experiment_file=parameters/coco/la_N1K5.yaml --sequential
python main.py --experiment_file=parameters/coco/la_N2K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/bam_N1K5.y --sequential
python main.py --experiment_file=parameters/pascal/bam_N2K5.y --sequential
python main.py --experiment_file=parameters/pascal/dcama_N1K5 --sequential
python main.py --experiment_file=parameters/pascal/dcama_N2K5 --sequential

# Ablations
python main.py --experiment_file=parameters/dcama_rank_test.yaml --sequential
python main.py --experiment_file=parameters/dcama_adapter_test.yaml --sequential
python main.py --experiment_file=parameters/dcama_encoder_test.yaml --sequential