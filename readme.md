
# RegScorer: Learning to select the best point cloud alignment for challenging scenarios

We consider the transformation selection task of identifying the optimal transformation from a set of candidates to register unaligned point clouds. This task is fundamental to successful point cloud registration but has long been overlooked. Existing approaches typically rely on traditional metrics, such as selecting the transformation with the highest inlier ratio. However, in challenging cases, such as symmetric scenes or low-overlap scenarios, local extrema transformations can also yield high inlier ratios, leading to suboptimal selections. To address this, we propose RegScorer that learns to score the candidate transformations for more reliable selection. RegScorer leverages both spatial and feature consistency of the aligned point clouds to learn patterns of correct and misleading transformations and score them, mitigating the influence of symmetry and local extrema. We evalute RegScorer on pairwise and multiview registration tasks, with a $\textbf{19.3/14.1}\%$ and $\textbf{4.7/5.1}\%$ recall boost on 3DLoMatch/ScanNet, respectively. When extended to the large-scale outdoor dataset WHU-TLS, RegScorer also achieves a $\textbf{25\%}$ improvement in registration recall. Also, the lightweight design ensures its high efficiency, requiring only 0.16 seconds per evaluation.

## Requirements

The code has been tested on:
- ubuntu 22.04 & ubuntu 18.04
- CUDA 11.1 & 12.1
- python 3.8.8 & python 3.10.13
- pytorch 1.8.1 & 2.2.1
- GeForce RTX 3070 & 4090

Please use the following command for installation:
```
pip install -r requirements.txt
python setup.py build develop
```

## Pretrained model

We provide pre-trained weights in the release page.
Please download the latest weights and put them in weights directory.

## Data preparation

For evaluate: 
    Put the source and target point clouds in /data/demo
    Put the candidate transformations in *.npz in /data/demo

## Evaluate

```
python score.py
```

Then, the score for each candidate transformation will be output in scores_*.npy


## Related Projects

We sincerely thank the excellent open-source projects:

- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [YOHO](https://github.com/HpWang-whu/YOHO)
- [FCGF](https://github.com/chrischoy/FCGF)
