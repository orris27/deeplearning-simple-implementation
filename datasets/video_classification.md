## 1. UCF101

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
