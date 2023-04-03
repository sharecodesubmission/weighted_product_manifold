cd ..

# run on small tree dataset.
# dim: number of hyperbolic space
# for hyp_dim in 2, 3, 4, 5, 7, 9, 10
python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 2 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 3 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 4 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 5 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 7 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 9 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/synthetic/smalltree.edges --batch-size 40 --dim 1 --hyp 10 -l 5.0 --epochs 100 --checkpoint-freq 100 --squareloss

