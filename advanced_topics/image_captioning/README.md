## How to use
1. Install pycoco first
```
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
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
