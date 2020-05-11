# SEMANTIC SEGMENTATION

## DATA 
```
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

packageID = 1 for gtFine
packageID = 3 for leftImg8bit
```
```bash
mkdir data 
mkdir data/train data/val data/test
mv gtFine leftImg8bit data
python data.py
python train.py
```
