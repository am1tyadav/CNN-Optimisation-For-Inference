# CNN-Optimisation-For-Inference

Many optimisation techniques can be applied to CNNs for faster inference and lower memory consumption:

1. Layer Fusion
2. Quantisation with or without Quantisation Aware Training
3. Weight Pruning (Sparsity) (To be done)
4. Weight Clustering (To be done)
5. Knowledge Distillation from a larger model to a smaller one (To be done)

and more..

This repo aims to demonstrate some of the above techniques.

## Usage

Create a virtual environment:

```bash
conda env update -f conda.yml

conda activate tf2
```

Available entry points:

```bash
python main.py --help
```
