## 1. UCF101
We can get the train/test split with the following script. Note that in `1_move_files.py`, the default split strategy is `01` version.
```bash
unzip UCF101_videos.zip 
cd UCF101
git clone https://github.com/harvitronix/five-video-classification-methods.git
mv five-video-classification-methods/data/* ./
rm -rf five-video-classification-methods/
sed -i "s/v_HandStandPushups/v_HandstandPushups/g" ucfTrainTestlist/*
python 1_move_files.py 
rm -rf 1_move_files.py 2_extract_files.py data_file.csv ucfTrainTestlist/
```

### annotation json
+ labels: `['ApplyEyeMakeup', 'ApplyLipstick', ...]`
+ databases:  dirname (contains several jpg frames for this video) => subset(training, validataion) + label
```python
p data['database']['v_WritingOnBoard_g03_c04']
{'subset': 'validation', 'annotations': {'label': 'WritingOnBoard'}}
```

With the above information, we can obtain the mapping from dirname (e.g. 'ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01') to its corresponding label (e.g. 'ApplyEyeMakeup'). A concrete example of the directory ('ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01') is shown below. For this video, we obtain 121 frames. "n_frames" is a ASCII file containing a float number which denotes the number of clips for this video. In this example, "n_frames" contains 121.
```python
['image_00001.jpg', 'image_00002.jpg', 'image_00003.jpg', 'image_00004.jpg', ... 'image_00120.jpg', 'image_00121.jpg', 'n_frames']
```

There are 9537 videos in UCF101.

The dataset we obtain is a list. An example is shown as below. `label` is the id for the class.
```python
segment = [1, n_frames]

[{'video': './datasets/UCF101_jpg/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01', 
'segment': [1, 121], 
'n_frames': 121, 
'video_id': 'v_ApplyEyeMakeup_g08_c01', 
'label': 0, 
'frame_indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]},
...]
```



### How to run [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs/)
1. Download videos [here](http://crcv.ucf.edu/data/UCF101.php). Suppose we get train/test split in the `datasets/UCF101/{train,test}`
2. Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py``` (Note that we do not need to split UCF101 to train/test, because `ucf101_0x.json` provides information for training or validation status)
```bash
mkdir -p datasets/UCF101_jpg/{train,test}
python utils/video_jpg_ucf101_hmdb51.py datasets/UCF101/train datasets/UCF101_jpg/train
python utils/video_jpg_ucf101_hmdb51.py datasets/UCF101/test datasets/UCF101_jpg/test
```

3. Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```
```bash
python utils/n_frames_ucf101_hmdb51.py datasets/UCF101_jpg/train
python utils/n_frames_ucf101_hmdb51.py datasets/UCF101_jpg/test
```

4. Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt
```bash
python utils/ucf101_json.py annotation_UCF101
sed -i "s/v_HandStandPushups/v_HandstandPushups/g" annotation_UCF101/*
```
5. Merge train/test split into one big folder (in my case, it is `UCF101_jpg`. Remember to delete `{train/test}` empty folder). Then run the model
```bash
python main.py --root_path ./ \
    --video_path datasets/UCF101_jpg/ \
    --annotation_path annotation_UCF101/ucf101_01.json \
    --result_path results \
    --dataset ucf101 \
    --n_classes 101 \
    --model mobilenet \
    --width_mult 0.5 \
    --train_crop random \
    --learning_rate 0.1 \
    --sample_duration 16 \
    --downsample 2 \
    --batch_size 64 \
    --n_threads 16 \
    --checkpoint 1 \
    --n_val_samples 1 \

```
