# CD-FSS
python main.py --experiment_file=parameters/cross/deepglobe.yaml --sequential
python main.py --experiment_file=parameters/cross/isic.yaml --sequential
python main.py --experiment_file=parameters/cross/lung.yaml --sequential

# FSS
python main.py --experiment_file=parameters/pascal/hdmnet_N1K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/hdmnet_N2K5.yaml --sequential
python main.py --experiment_file=parameters/coco/la_N1K5.yaml --sequential
python main.py --experiment_file=parameters/coco/la_N2K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/bam_N1K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/bam_N2K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/dcama_N1K5.yaml --sequential
python main.py --experiment_file=parameters/pascal/dcama_N2K5.yaml --sequential
python main.py --experiment_file=parameters/coco/fptrans_N1K5.yaml
python main.py --experiment_file=parameters/coco/fptrans_N2K5.yaml
python main.py --experiment_file=parameters/pascal/fptrans_N1K5.yaml
python main.py --experiment_file=parameters/pascal/fptrans_N2K5.yaml

# Ablations
python main.py --experiment_file=parameters/dcama_rank_test.yaml --sequential
python main.py --experiment_file=parameters/dcama_adapter_test.yaml --sequential
python main.py --experiment_file=parameters/dcama_encoder_test.yaml --sequential

# Adaptive
python main.py --experiment_file=parameters/coco/dcama_ada_N1K5.yaml
python main.py --experiment_file=parameters/coco/dcama_ada_N2K5.yaml
python main.py --experiment_file=parameters/pascal/dcama_ada_N1K5.yaml
python main.py --experiment_file=parameters/pascal/dcama_ada_N2K5.yaml
python main.py --experiment_file=parameters/coco/fptrans_ada_N1K5.yaml
python main.py --experiment_file=parameters/coco/fptrans_ada_N2K5.yaml
python main.py --experiment_file=parameters/pascal/fptrans_ada_N1K5.yaml
python main.py --experiment_file=parameters/pascal/fptrans_ada_N2K5.yaml

# Decoder
python main.py --experiment_file=parameters/decoder/dcama_N1K5.yaml
python main.py --experiment_file=parameters/decoder/dcama_N2K5.yaml

python main.py --experiment_file=parameters/decoder/fptrans_N1K5.yaml
python main.py --experiment_file=parameters/decoder/fptrans_N2K5.yaml

python main.py --experiment_file=parameters/decoder/bam_N1K5.yaml
python main.py --experiment_file=parameters/decoder/bam_N2K5.yaml
python main.py --experiment_file=parameters/decoder/hdmnet_N1K5.yaml
python main.py --experiment_file=parameters/decoder/hdmnet_N2K5.yaml
python main.py --experiment_file=parameters/decoder/la_N1K5.yaml
python main.py --experiment_file=parameters/decoder/la_N2K5.yaml

# Computational
python main.py --experiment_file=parameters/computation/hdmnet.yaml
python main.py --experiment_file=parameters/computation/bam.yaml
python main.py --experiment_file=parameters/computation/la.yaml
python main.py --experiment_file=parameters/computation/dcama.yaml
python main.py --experiment_file=parameters/computation/fptrans.yaml
python main.py --experiment_file=parameters/computation/dcama_ada.yaml
python main.py --experiment_file=parameters/computation/fptrans_ada.yaml

python main.py --experiment_file=parameters/computation/decoder/hdmnet.yaml
python main.py --experiment_file=parameters/computation/decoder/bam.yaml
python main.py --experiment_file=parameters/computation/decoder/la.yaml
python main.py --experiment_file=parameters/computation/decoder/dcama.yaml
python main.py --experiment_file=parameters/computation/decoder/fptrans.yaml