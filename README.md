# Ternary-Structured-Receptive-Field
This is a Pytorch implementation of [Structured-Receptive-Fields-in-CNNs](https://arxiv.org/pdf/1605.02971v2.pdf) combined with [Ternary-Weights-Network](https://arxiv.org/abs/1605.04711) for the MNIST dataset. It is the first time both concepts have been combined, to my knowledge. The dataset is provided by [torchvision](https://pytorch.org/docs/master/torchvision/). There are 2 files: `main.py` - for a regular ternarized LeNet model - and `second_main.py` - for a ternarized Structured Receptive Field model.

# Requirements
- Python, Numpy
- Pytorch 0.3.1

# Usage

    $ git clone https://github.com/buaabai/Ternary-Weights-Network
    $ python main.py --epochs 100
    $ python second_main.py --epochs 100

You can use

    $ python main.py -h

to check other parmeters.




