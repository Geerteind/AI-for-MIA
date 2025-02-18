# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:03:12 2025

@author: 20212287
"""
import numpy as np
four_classes = {}
four_classes["vertical digits"] = [1, 7]
four_classes["loopy digits"] = [0, 6, 8, 9]
four_classes["curly digits"] = [2, 5]
four_classes["other"] = [3, 4]

y = np.array([1,7,0,6,8,9,2,5,3,4])
return_array = y


for a in y:
    for key, values in four_classes.items():
        if a in values:
            returned_value = key
            #print(four_classes.items())
            return_array = np.where(y == a, returned_value, return_array)


