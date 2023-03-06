# -*- coding: utf-8 -*-
"""
The core module of uncertainty analysis
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from scipy.optimize import minimize#, basinhopping, brute

# global variable
Nfeval = 1

def calib(func, X, Y, theta0, sig2_y0, Niter=5, bnds=None):
    ''' Function for parameter calibration (estimation)
    Args:
        func -- the function to predict Y given X
        X, Y -- input/output data
        theta0 -- initial parameter values
        sig2_y0 -- initial guess of output variance
    Rtns:
    '''
    if Y.shape[1] > 1:
        raise ValueError('Invalid Y.shape[1] which currently must be 1')

    theta = np.empty_like(theta0)
    sig2_y = np.empty_like(sig2_y0)
    np.copyto(theta, theta0)
    np.copyto(sig2_y, sig2_y0)
    
    for i in range(Niter):
        feval = calc_theta_obj(theta, func, X, Y, sig2_y)
        print('Iter {0:4d}, f = {1:.6f}'.format(i, float(feval[0])))

        if bnds == None:
            res = minimize(calc_theta_obj, theta, args=(func, X, Y, sig2_y), 
                           method='BFGS', options={'disp': True, 'maxiter': 100})
        else:
            res = minimize(calc_theta_obj, theta, args=(func, X, Y, sig2_y), 
                       method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxiter': 100})
        
        np.copyto(theta, res.x)
        sig2_y = calc_sig2y(theta, func, X, Y)        

    # calculate the hessian to determine the variability of the parameter estimate
    H = calcHessVargs(calc_theta_obj, theta, func, X, Y, sig2_y)
    # print theta, H
    V = np.linalg.inv(H)
    
    return (theta, sig2_y, V)

    
def calc_theta_obj(theta, func, X, Y, sig2_y):
    ''' The objective function (negative log-likelihood) for optimising theta
    Terms that are not dependent on theta (thus constant as far as optimisation is concerned) are not calculated.
    '''
    alpha = 1.0/sig2_y    
    y_pred = func(theta, X)
    err = y_pred - Y
    neg_lnlik = 0.5*(alpha* np.sum(err**2))
    return neg_lnlik

        
def pred(func, X, theta, V, sig2_y=0):
    ''' Function to make prediction (both mean and variance
    Args:
    theta - parameter (estimated)
    V - the covariance matrix of theta
    sig2_y - the covariance of output y to be predicted
    '''

    n_dat = X.shape[0]
    #n_theta = theta.shape[0]
       
    Y_mean = func(theta, X)
    grad = calc_grad_theta(func, theta, X) # grad in shape of [n_dat, n_paras]
        
    Y_cov = np.zeros(n_dat)
    for i in range(n_dat): 
        gd = np.mat(grad[i,:])
        #print gd
        Y_cov[i] = gd * V * np.matrix.transpose(gd) + sig2_y
        #print Z

    return (Y_mean, Y_cov)   

def calc_sig2y(theta, func, X, Y):
    ''' The function to calculate the optimal values of the variance term
    '''
    
    n_dat = X.shape[0]

    sse = .0
    sig2_y = .0
    
    for n in range(n_dat):        
        y_pred = func(theta, X[n,:])
        err = Y[n] - y_pred
        sse += err**2
        print(sse)
    sig2_y = sse / n_dat
    
    return sig2_y

    
def calc_grad_theta(func, theta, X):
    '''Calculate the gradient of <func> w.r.t parameters <theta> with given <X>
    using finite difference
    '''
    n_theta = theta.shape[0]
    n_dat = X.shape[0]

    f = func(theta, X)
    
    theta1 = np.zeros(theta.shape)
    grad = np.zeros((n_dat, n_theta))

    for i in range(n_theta):
        delta = theta[i]*1e-4
        np.copyto(theta1,theta)
        if np.abs(delta) < 1e-5:
            delta = 1e-5
        theta1[i] += delta

        f = func(theta, X)
        f1 = func(theta1, X)

        gd = (f1-f) / delta
        grad[:,i] = np.squeeze(gd)

    return grad

    
###########################################################
def calcHessVargs(func_post, paras, *args):
    ''' Function to calculate the Hessian of negative
        log posterior/likelihood w.r.t. model parameters with variable arguments
    '''

    n_paras = len(paras)
    H = np.zeros( (n_paras, n_paras) )
    
    delta_paras = np.fabs(paras) * 1e-3
    delta_paras[ delta_paras<1e-8 ] = 1e-8 # to avoid too small values

    for i in range(n_paras):
        for j in range(n_paras):

            if (i>j):
                H[i,j] = H[j,i]

            else:
                p1 = np.copy(paras)                
                p1[i] += delta_paras[i]
                p1[j] += delta_paras[j]
                t1 = func_post(p1, *args)

                p2 = np.copy(paras)                
                p2[i] += delta_paras[i]
                p2[j] -= delta_paras[j]
                t2 = func_post(p2, *args)

                p3 = np.copy(paras)                
                p3[i] -= delta_paras[i]
                p3[j] += delta_paras[j]
                t3 = func_post(p3, *args)

                p4 = np.copy(paras)                
                p4[i] -= delta_paras[i]
                p4[j] -= delta_paras[j]
                t4 = func_post(p4, *args)

                H[i,j] = (t1-t2-t3+t4) / (4*delta_paras[i]*delta_paras[j])            

    return H



###########################################################
def lognormpdf(x,mu,S):
    """ Calculate Gaussian probability density of x, when x ~ N(mu,sigma) """

    nx = len(S)
    norm_coeff = nx*np.log(2*np.pi)+np.linalg.slogdet(S)[1]

    err = x-mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return -0.5*(norm_coeff+numerator)