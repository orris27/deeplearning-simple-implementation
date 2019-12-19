#### 1. Outputs of VGG16 are nearly same even if we randomly sample the inputs?
1. Run the following codes and we can observe that the outputs of the last 2 samples are roughly same. Since the 1st sample is completely zero, and the 2nd sample is ones, they are slightly different from the 3rd and 4th samples.
2. Besides, if I use this `features` to train CIFAR10, the validation accuracy is no good than 10%. However, if I use the vgg16 in pytorch, the validation accuracy can be higher than 80%.
```python
import torch
import torch.nn as nn


def make_features(channels, depth=13):
    layers = []
    for i in range(depth):
        layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))
        layers.append(nn.ReLU())
        if i in [1, 3, 6, 9, 12]:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    #print(layers)
    return nn.Sequential(*layers)


channels = [3, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

for i in range(13):
    print('%d-th layer:'%(i), end=' ')
    model = make_features(channels, i)
    model = model.eval()
    x = torch.randn(4, 3, 32, 32)
    x[0, :, :, :] = 0
    x[1, :, :, :] = 1
    print(list(map(float, model(x).squeeze().sum(1).detach().numpy())))
```
The reason is that I do not initialize the parameters of VGG16 by myself. Check [here](https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py) for more information.
