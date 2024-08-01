# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:51:13 2017

@author: tc0008
"""
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt

from uncertainty import unctn
reload(unctn)
from qspr import stracorn_lp
reload(stracorn_lp)

dat_Dlp = np.loadtxt("qspr/MW_Dlp.txt")

paras0 = np.log( np.array([2.54e-5, 0.456]) )
#bnds = ((-10, 10), (-10, 10))
bnds = None
sig2_y = np.array([0.05])

X = dat_Dlp[:,0].reshape((-1, 1))
Y = np.log( dat_Dlp[:,1].reshape((-1,1)) )

rlt = unctn.calib(stracorn_lp.compD_ln, X, Y, paras0, sig2_y, 5, bnds)
paras, sig2_y, V = rlt
prd = unctn.pred(stracorn_lp.compD_ln, X, paras, V, sig2_y)

r_mw = np.power( 0.91*dat_Dlp[:,0]/4*3/np.pi, 1.0/3 )

plt.plot(r_mw, np.squeeze(prd[0]), 'x', r_mw, Y, 'o')
#plt.plot(np.log10(Xy[:,0]), np.squeeze(rlt_pred0[2]), '^')
plt.show(0)