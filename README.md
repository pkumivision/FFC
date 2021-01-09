# Fast Fourier Convolution (FFC) for Image Classification

This is the official code of [Fast Fourier Convolution](https://papers.nips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html) for image classification on ImageNet.

## Quick starts
### Requirements

- pip install -r requirements.txt

### Data preparation
You can follow the Pytorch implementation:
https://github.com/pytorch/examples/tree/master/imagenet

### Training

To train a model, run [main.py](main.py) with the desired model architecture and other super-paremeters:

```bash
python main.py -a ffc_resnet50 --lfu [imagenet-folder with train and val folders]
```

We use "lfu" to control whether to use Local Fourier Unit (LFU). Default: False. 

### Testing
```bash
python main.py -a ffc_resnet50 --lfu --resume PATH/TO/CHECKPOINT [imagenet-folder with train and val folders]
```

## Citation
If you find this work or code is helpful in your research, please cite:
````
@InProceedings{Chi_2020_CVPR,
  author = {Chi, Lu and Yuan, Zehuan and Mu, Yadong and Wang, Changhu},
  title = {Non-Local Neural Networks With Grouped Bilinear Attentional Transforms},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
````