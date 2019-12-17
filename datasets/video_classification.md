## 1. UCF101
We can get the train/test split with the following script. Note that in `1_move_files.py`, the default split strategy is `01` version.
```
unzip UCF101_videos.zip 
cd UCF101
git clone https://github.com/harvitronix/five-video-classification-methods.git
mv five-video-classification-methods/data/* ./
rm -rf five-video-classification-methods/
sed -i "s/v_HandStandPushups/v_HandstandPushups/g" ucfTrainTestlist/*
python 1_move_files.py 
rm -rf 1_move_files.py 2_extract_files.py data_file.csv ucfTrainTestlist/
```
### How to run [Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs/)
1. Download videos [here](http://crcv.ucf.edu/data/UCF101.php). Suppose we get train/test split in the `datasets/UCF101/{train,test}`
2. Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py``` (Note that we do not need to split UCF101 to train/test, because `ucf101_0x.json` provides information for training or validation status)
```bash
mkdir -p datasets/UCF101_jpg/{train, test}
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
5. Run the model
```bash
python main.py --root_path ./ \
	--video_path datasets/UCF101_jpg/train \
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
