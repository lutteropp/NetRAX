#!/bin/bash
for i in *.csv
do  
   python3 brlen_histograms.py $i
done
