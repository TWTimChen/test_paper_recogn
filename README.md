# Setup
```
cd ./../test_paper_recogn
pip install -r requirements.txt
```

# Manual
## [seg.py](../seg.py)
> Usage: Water paper segmentation. 
> 
> -i  [IMAGE] 	path to the input image
> -s			show image for each steps
> -w			store the proccessed wrap image

```
python seg.py -s -i ./img/IMG_20190621_094138.jpg
```
## [counter.py](../counter.py)
> Usage: Detect spots & extract infomation.
> 
> -i  [IMAGE] 	path to the input image
> -s			show image for each steps
> -w			store spot_info to csv file
```
python counter.py -s -i ./img/IMG_20190621_094138_seg.jpg
```
## [stat.py](../stat.py)
> Usage: descriptive statistics for spots info
> 
> -i  [DATA] 	path to the input csv
```
python stat.py -i ./data/IMG_20190621_094138_seg.csv 
```
