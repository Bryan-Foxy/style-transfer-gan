#!/bin/bash

mkdir "../datas"
cd "../datas"
curl -L -o datasets.zip https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/monet2photo.zip
unzip datasets.zip
rm datasets.zip
cd ../