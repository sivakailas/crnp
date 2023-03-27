#!/bin/bash
for i in `seq 0 10`
do
   python main.py --logid 2023-02-27-04-20-33-227982 --runid $i --epoch -1
done