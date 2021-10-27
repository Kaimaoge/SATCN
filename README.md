# SATCN
We present SATCN---Spatial Aggregation and Temporal Convolution Networks---a universal and flexible framework to perform spatiotemporal kriging for various spatiotemporal datasets without the need for model specification. Specifically, we propose a novel spatial aggregation network (SAN) inspired by Principal Neighborhood Aggregation, which uses multiple aggregation functions to help one node gather diverse information from its neighbors. To exclude information from unsampled nodes, a masking strategy that prevents the unsampled sensors from sending messages to their neighborhood is introduced to SAN. We capture temporal dependencies by the temporal convolutional networks, which allows our model to cope with data of diverse sizes. To make SATCN generalizable to unseen nodes and even unseen graph structures, we employ an inductive strategy to train SATCN.

## Model framework

<img src="https://github.com/Kaimaoge/SATCN/blob/main/fig/whole_framework-1.png" width="800">

## Paper
[Wu, Y., Zhuang, D., Lei, M., Labbe, A., & Sun, L. (2021). Spatial Aggregation and Temporal Convolution Networks for Real-time Kriging. arXiv preprint arXiv:2109.12144.](https://arxiv.org/pdf/2109.12144.pdf)
