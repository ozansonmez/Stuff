#!/usr/bin/python
import os
import math
import sys

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()

    x=map(float,words)[0]
    y=map(float,words)[1]

    x_lo = math.floor(x*10)/10
    x_high = math.ceil(x*10)/10
    
    y_lo = math.floor(y*10)/10
    y_high = math.ceil(y*10)/10

    print '%s\t%s' % ((x_lo,x_high,y_lo,y_high), 1)




