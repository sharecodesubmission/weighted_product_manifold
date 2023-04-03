cd ..
python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 5 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 10 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 20 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 30 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 50 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 70 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
# python pytorch/pytorch_hyperbolic.py learn data/edges/ca-CSphd.edges --batch-size 1024 --dim 1 --hyp 100 -l 10.0 --epochs 5000 --checkpoint-freq 100 --subsample 16 --squareloss
