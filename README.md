## Hierarchical Representations with Poincaré Variational Auto-Encoders

This repo contains reimplementation of the models presented in
[Hierarchical Representations with Poincaré Variational Auto-Encoders](https://arxiv.org/abs/1901.06033)

`@misc{1901.06033, Author = {Emile Mathieu and Charline Le Lan and Chris J. Maddison and Ryota Tomioka and Yee Whye Teh}, Title = {Hierarchical Representations with Poincaré Variational Auto-Encoders}, Year = {2019}, Eprint = {arXiv:1901.06033}, }`

## Toy model

To train the model on the toy dataset run

```
python main.py --dataset toy --distribution wrapped --prior_sigma 1.7 --epochs 1000
```

## MNIST

Training the model on MNIST

```
python main.py --dataset mnist --distribution riemannian --prior_sigma 1.7 --epochs 1000 --break-early 1 --break-interval 30 --batch_size 128 --test_batch 128
```
