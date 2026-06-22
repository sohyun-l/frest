# Source model weights

Place the Cityscapes-pretrained SegFormer-B5 checkpoint here:

```
pretrained_models/segformer.b5.1024x1024.city.160k.pth
```

This is the official SegFormer-B5 model trained on Cityscapes (1024x1024, 160k),
used as the source model for adaptation. Download it from the official SegFormer
release:

- https://github.com/NVlabs/SegFormer  (weights: `segformer.b5.1024x1024.city.160k.pth`)

The training and evaluation scripts expect the file at the exact path above.

## Alignment weights (UAWarpC, MegaDepth)

Training warps the reference images with a UAWarpC matching network. Place its
MegaDepth-pretrained weights here:

```
pretrained_models/uawarpc_megadepth.ckpt
```

Available from the Refign/CMA release: https://github.com/brdav/cma
(These are only needed for training; evaluation does not use them.)
