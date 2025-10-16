# https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

cd ..
cd ..
cd data
mkdir mscoco
cd mscoco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

cd ../
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip