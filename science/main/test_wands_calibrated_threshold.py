#!/usr/bin/env python3
import numpy as np
from run_wands_calibrated_lexical import calibrate
x=calibrate(np.array([.9,.9,.8,.7]),np.array([1,0,1,0]))
assert x['threshold']==.8 and x['tp']==2 and x['predicted_positive']==3 and x['f1']==.8
x=calibrate(np.array([.9,.8,.7,.6]),np.array([1,0,0,1]))
assert x['threshold']==.9 and x['tp']==1 and x['predicted_positive']==1 and x['f1']==2/3
print('F1 threshold selection and highest-threshold tie handling verified.')
