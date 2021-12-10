#!/bin/bash



python eval_cosim.py --loss arcface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.3
python eval_cosim.py --loss arcface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 10 --thr 0.3

#python eval_cosim.py --loss arcface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.4
#python eval_cosim.py --loss arcface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.35
#python eval_cosim.py --loss arcface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.375


#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 10 --thr 0.4
#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 10 --thr 0.3
#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 10 --thr 0.35

#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.4
#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.3
#python eval_cosim.py --loss cosface --n_iter_intra 500 --n_iter_inter 5000 --n_frame 5 --thr 0.35
