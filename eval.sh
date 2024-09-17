for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python model_eval.py --model_path vit-base-patch16-224 --sparse_ratio $ratio --compress_ratio 2
done