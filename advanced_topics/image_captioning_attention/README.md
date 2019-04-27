## How to use
1. Install pycoco first
```
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install

chmod +x download.sh
./download.sh
```

2. Preprocess data
```
python build_vocab.py # produce vocab.pkl file
python resize.py # resize the image for ResNet
```
3. Train the model
```
python train.py
```

4. Evaluate model
```
python sample.py
```

## Inference
![giraffe](./data/giraffe.png)

a giraffe standing in a field next to a tree .

![surf](./data/surf.jpg)

a man riding a wave on top of a surfboard .
