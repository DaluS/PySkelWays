# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:28:47 2022

@author: Paul Valcke
"""


import pyskelways as psw
import sys
path = "C:\\Users\\Paul Valcke\\Documents\\GitHub\\PySkelWays"  # Where pygemmes is
sys.path.insert(0, path)  # we tell python to look at the folder `path`

img_address = ''
dparams = {}

hub = psw.Hub()
hub
hub.set_dparam(dparams)

# INDIVIDUAL EXECUTION
hub.Load_Image(img_address)
hub.Compute_binarisation()
hub.Compute_contours()
hub.Compute_arcs()
hub.Compute_places()
hub.Compute_ways()

# COLLECTIVE EXECUTION
hub.Compute_all(img_address)
