''' The module that contains computational routines for hybrid model
identification and uncertainty quantification
'''

import warnings
import numpy as np
#import math
#import scipy.sparse as sp
#import scipy.sparse.linalg as spln
#import matplotlib.pyplot as plt
from scipy.optimize import minimize#, basinhopping, brute

from uncertainty.unctn import calcHessVargs, lognormpdf

# global variable
Nfeval = 1


###########################################################
def PluginMain(func_top, func_low, Xy, Y, Xz, Z, theta0, sig2_y0, sig2_z0, Niter=10, bnds=None):
    ''' The main function to run plug-in algorithm for parameter estimation
    Args:
    Rtns:
    '''

    theta = np.empty_like(theta0)
    sig2_y = np.empty_like(sig2_y0)
    sig2_z = np.empty_like(sig2_z0)
    np.copyto(theta, theta0)
    np.copyto(sig2_y, sig2_y0)
    np.copyto(sig2_z, sig2_z0)

    global Nfeval
    Nfeval = 1 
    
    for i in range(Niter):
        feval = Plugin_theta_obj(theta, func_top, func_low, Xy, Y, Xz, Z, sig2_y, sig2_z)
        print('Plugin Iter {0:4d}, f = {1:.6f}'.format(i, float(feval[0])))

        if bnds == None:
            res = minimize(Plugin_theta_obj, theta, args=(func_top, func_low, Xy, Y, Xz, Z, sig2_y, sig2_z), 
                           method='BFGS', callback=callbackF, options={'disp': True, 'maxiter': 100})
        else:
            res = minimize(Plugin_theta_obj, theta, args=(func_top, func_low, Xy, Y, Xz, Z, sig2_y, sig2_z), 
                       method='L-BFGS-B', bounds=bnds, callback=callbackF, options={'disp': True, 'maxiter': 100})
        
        np.copyto(theta, res.x)
        var = Plugin_var(theta, func_top, func_low, Xy, Y, Xz, Z)
        np.copyto(sig2_y, var[0])
        np.copyto(sig2_z, var[1])

    # calculate the hessian to determine the variability of the parameter estimate
    H = calcHessVargs(Plugin_theta_obj_wrapper, theta, func_top, func_low, Xy, Y, Xz, Z, sig2_y, sig2_z)
    # print theta, H
    V = np.linalg.inv(H)
    
    return (theta, sig2_y, sig2_z, V)

def Plugin_theta_obj_wrapper(theta, *args):
    ''' The wrapper to pass variable arguments to Plugin_theta_obj
    '''
    return Plugin_theta_obj(theta, *args)

    
def Plugin_theta_obj(theta, func_top, func_low, Xy, Y, Xz, Z, sig2_y, sig2_z):
    ''' The objective function (negative log-likelihood) for optimising theta using the plug-in method
    Terms that are not dependent on theta (thus constant as far as optimisation is concerned)
    are not calculated.
    '''
   
    n_dat_Xy = Xy.shape[0]
    
    beta = 1.0/sig2_z
    alpha = 1.0/sig2_y

    neg_lnlik = .0
    
    # For high-level X-Y data
    for n in range(n_dat_Xy):
        #print '\t theta = {0:}'.format(theta)
        y_func = func_top(theta, Xy[n,:].reshape((1,-1)), func_low)
        err = Y[n,:] - y_func
        neg_lnlik += 0.5*(alpha*err*err)

    # For low-level X-Z data
    d_z = len(Xz)

    # Data in Xz and Z are saved in lists, each item in the lists represent one
    #    Z-variable
    for i in range (d_z):
        Xtmp = Xz[i]
        Ztmp = Z[i]
        n_dat_Xz = Xtmp.shape[0]   
        for n in range(n_dat_Xz):
            z_func = func_low( theta, Xtmp[n,:].reshape((1,-1)) )
            err = Ztmp[n] - z_func[:,i]
            #print '\t err = {0:}'.format(err)
            neg_lnlik += 0.5* np.sum( beta[i]* (np.array(err)**2) )
            
    if np.isfinite(neg_lnlik):
        return neg_lnlik
    else:
        return 1e10

def Plugin_var(theta, func_top, func_low, Xy, Y, Xz, Z):
    ''' The function to calculate the optimal values of the variance terms using the plug-in method
    '''
    
    n_dat_Xy = Xy.shape[0]

    sse_l = np.zeros(len(Z))
    sse_h = np.zeros(Y[0,:].shape)
    sig2_z = np.zeros(sse_l.shape)
    sig2_y = np.zeros(sse_h.shape)
    
    # For high-level X-Y data
    for n in range(n_dat_Xy):
        #print Xy[n,:].reshape((1,-1))
        y_func = func_top(theta, Xy[n,:].reshape((1,-1)), func_low)
        err = Y[n,:] - y_func
        sse_h += np.squeeze(err)**2
    sig2_y = sse_h / n_dat_Xy
    
    # For low-level X-Z data
    d_z = len(Xz)

    # Data in Xz and Z are saved in lists, each item in the lists represent one
    #    Z-variable
    for i in range (d_z):
        Xtmp = Xz[i]
        Ztmp = Z[i]
        n_dat_Xz = Xtmp.shape[0]   
        for n in range(n_dat_Xz):
            z_func = func_low( theta, Xtmp[n,:].reshape((1,-1)) )
            err = Ztmp[n] - z_func[:,i]
            #print sse_l[i].shape
            #print err.shape
            sse_l[i] += err*err

        sig2_z[i] = sse_l[i] / (n_dat_Xz)

    return (sig2_y, sig2_z)

###########################################################
def EMmain(func_top, func_low, Xy, Y, Xz, Z, theta0, sig2_y0, sig2_z0, Nmc=10, Niter=10, bnds=None):
    ''' The main function to run EM algorithm for parameter estimation
    Args:
    Rtns:
    '''

    theta = np.empty_like(theta0)
    sig2_y = np.empty_like(sig2_y0)
    sig2_z = np.empty_like(sig2_z0)
    np.copyto(theta, theta0)
    np.copyto(sig2_y, sig2_y0)
    np.copyto(sig2_z, sig2_z0)
    
    for i in range(Niter):
        Zsamples = Estep(func_top, func_low, theta, Xy, Y, sig2_y, sig2_z, Nmc)
        feval = Mstep_theta_obj(theta, func_top, func_low, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z)
        print( 'EM Iter {0:4d}, f = {1:.6f}'.format(i, float(feval[0])) )
        
        Mrlt = Mstep_main(func_top, func_low, theta, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z, bnds)
        np.copyto(theta, Mrlt[0])
        np.copyto(sig2_y, Mrlt[1])
        np.copyto(sig2_z, Mrlt[2])

    # calculate the hessian to determine the variability of the parameter estimate
    H = calcHessVargs(Mstep_theta_obj_wrapper, theta, func_top, func_low, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z)
    V = np.linalg.inv(H)
    
    return (theta, sig2_y, sig2_z, V, Zsamples)


###########################################################
def Estep(func_top, func_low, theta, X, Y, sig2_y, sig2_z, N=100):
    ''' The E-step of the Monte Carlo EM algorithm
    Args:
    - func_top: top level function;  y = f(theta,x,z) + epsilon
    - func_low: low level function;  z = h(theta,x)   + zeta
    - theta: model parameters
    - X: data for input variables
    - Y: data for top level output variables
    - sig2_y: variance of top level function noise (epsilon)
    - sig2_z: variance of low level function noise (zeta)
    - N: number of MC samples
    Rtns:
    '''

    n_dat = X.shape[0]
    d_y = Y.shape[1]
    d_z = 2 #func_low(theta, np.array(X[0,:])).shape[0] # dimension of z

    Ymc = np.zeros((N, d_y))
    Z = np.zeros((N, d_z, n_dat)) # MC samples
    W = np.zeros((N, n_dat)) # weights

    sig_y = np.sqrt(sig2_y)
    sig_z = np.sqrt(sig2_z)

    for n in range(n_dat):

        z_n = func_low(theta, X[n,:].reshape(1,-1))        
        Z[:,:,n] = np.tile(z_n,(N,1))
        np.tile(z_n,(N,1))

        # MC sampling
        for d in range(d_z):
            rd = np.random.normal(0, sig_z[d], N)
            Z[:,d,n] += rd

        s = .0;
        for i in range(N):
            Ymc[i,:] = func_top( theta, np.array(X[n,:]), Z[i,:,n] )
            aa = np.exp( lognormpdf( Ymc[i,:], Y[n,:], np.diag(sig2_y) ) )
            if np.isfinite(aa):
                W[i,n] = aa
            else:
                W[i,n] = 0
            s += W[i,n]
        W[:,n] /= s # normalising the weights

    return (Z, W)


###########################################################
def Mstep_main(func_top, func_low, theta0, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z, bnds=None):
    ''' Function to run the M-step
    '''
    
    global Nfeval
    Nfeval = 1

    if bnds == None:
        res = minimize(Mstep_theta_obj, theta0, args=(func_top, func_low, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z), 
                       method='BFGS', callback=callbackF, options={'disp': True, 'maxiter': 100})
    else:
        res = minimize(Mstep_theta_obj, theta0, args=(func_top, func_low, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z), 
                       method='L-BFGS-B', bounds=bnds, callback=callbackF, options={'disp': True, 'maxiter': 100})
   
    theta1 = res.x
    var = Mstep_var(theta1, func_top, func_low, Xy, Y, Zsamples, Xz, Z)
    
    return (theta1, var[0], var[1])



def callbackF(Xi):
    ''' callback function to display information for the optimiser
    '''
    global Nfeval
    print( '\t Iter {0:4d}:  Para values: {1:}'.format(Nfeval, Xi) )
    Nfeval += 1

def Mstep_theta_obj_wrapper(theta, *args):
    ''' The wrapper to pass variable arguments to Mstep_theta_obj
    '''
    return Mstep_theta_obj(theta, *args)
    #return Mstep_theta_obj(theta, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8])

    
def Mstep_theta_obj(theta, func_top, func_low, Xy, Y, Zsamples, Xz, Z, sig2_y, sig2_z):
    ''' The objective function (negative log-likelihood) for optimising theta
    Terms that are not dependent on theta (thus constant as far as optimisation is concerned)
    are not calculated.
    '''
   
    n_dat_Xy = Xy.shape[0]
    Zmc = Zsamples[0]
    Wmc = Zsamples[1]    
    n_mc = Zmc.shape[0]
    
    beta = 1.0/sig2_z
    alpha = 1.0/sig2_y

    neg_lnlik = .0
    
    # For high-level X-Y data
    for n in range(n_dat_Xy):
        #print '\t theta = {0:}'.format(theta)
        z_func = func_low( theta, Xy[n,:].reshape((1,-1)) )

        for i in range(n_mc):
            err = Zmc[i,:,n] - z_func
            neg_lnlik += Wmc[i,n]* 0.5* np.sum( beta* (np.array(err)**2) )
            y_func = func_top(theta, Xy[n,:], Zmc[i,:,n])
            err = Y[n,:] - y_func
            neg_lnlik += Wmc[i,n]* 0.5*(alpha*err*err)

    # For low-level X-Z data
    d_z = len(Xz)

    # Data in Xz and Z are saved in lists, each item in the lists represent one
    #    Z-variable
    for i in range (d_z):
        Xtmp = Xz[i]
        Ztmp = Z[i]
        n_dat_Xz = Xtmp.shape[0]   
        for n in range(n_dat_Xz):
            z_func = func_low( theta, Xtmp[n,:].reshape((1,-1)) )
            err = Ztmp[n] - z_func[:,i]
            #print '\t err = {0:}'.format(err)
            neg_lnlik += 0.5* np.sum( beta[i]* (np.array(err)**2) )
            
    if np.isfinite(neg_lnlik):
        return neg_lnlik
    else:
        return 1e10

## not verified yet
def Mstep_theta_grad(func_top, func_low, theta, X, Y, sig2_y, sig2_z, Z, W):
    ''' The gradient of the objective function with respect to theta
    '''
    n_dat = X.shape[0]
    d_y = Y.shape[1]
    n_mc = Z.shape[0]
    d_z = Z.shape[1]

    beta = 1.0/sig2_z
    alpha = 1.0/sig2_y

    grad = np.zeros(theta.shape)
    for n in range(n_dat):
        
        z_func = func_low(theta, X[n,:])
        y_func = func_top(theta, X[n,:], Z[:,:,n])
        gd_h = calc_grad(func_low, theta, X[n,:])

        for i in range(n_mc):
            err = Z[i,:,n] - z_func
            gd1 = beta*err* gd_h
            err = Y[n,:] - y_func
            gd2 = alpha*err* calc_grad(func_top, theta, X[n,:], Z[i,:,n])
            grad += W[i,n]* (gd1+gd2)

    return grad

def calc_grad_theta_Z(func, theta, X, Z):
    '''Calculate the gradient of <func> w.r.t parameters <theta> & <Z>, with given <X> and <Z>
    using finite difference
    '''

    grad_theta = calc_grad_theta(func, theta, X, Z)
#todo: check this
    n_Z = Z.shape[1]
    n_dat = X.shape[0]

    f = func(theta, X, Z)
    if f.ndim == 1:
        d_f = 1
    else:
        d_f = f.shape[1]
    
    Z1 = np.zeros(Z[0,:].shape)
    grad = np.zeros((n_dat, d_f, n_Z))

    for i in range(n_dat):
        for j in range(n_Z):

            delta = Z[i,j]*1e-4
            np.copyto(Z1, Z[i,:])
            if np.abs(delta) < 1e-5:
                delta = 1e-5
            Z1[j] += delta

            f = func(theta, X[i,:].reshape((1,-1)), Z[i,:].reshape((1,-1)))
            f1 = func(theta, X[i,:].reshape((1,-1)), Z1.reshape((1,-1)))

            gd = (f1-f) / delta
            if np.isscalar(gd):
                grad[i,:,j] = gd
            else:
                grad[i,:,j] = gd[:,None]

    return (grad_theta, grad)

def calc_grad_theta(func, theta, X, Z=None):
    '''Calculate the gradient of <func> w.r.t parameters <theta> with given <X> and <Z>
    using finite difference
    '''
    n_theta = theta.shape[0]
    n_dat = X.shape[0]

    if Z is not None:
        f = func(theta, X, Z)
    else:
        f = func(theta, X)

    if f.ndim == 1:
        d_f = 1
    else:
        d_f = f.shape[1]
    
    theta1 = np.zeros(theta.shape)
    grad = np.zeros((n_dat, d_f, n_theta))

    for i in range(n_theta):
        delta = theta[i]*1e-4
        np.copyto(theta1,theta)
        if np.abs(delta) < 1e-5:
            delta = 1e-5
        theta1[i] += delta

        if Z is not None:
            f = func(theta, X, Z)
            f1 = func(theta1, X, Z)
        else:
            f = func(theta, X)
            f1 = func(theta1, X)

        gd = (f1-f) / delta
        if gd.ndim < 2:
            grad[:,:,i] = gd[:,None]
        else:
            grad[:,:,i] = gd

    return grad


def Mstep_var(theta, func_top, func_low, Xy, Y, Zsamples, Xz, Z):
    ''' The function to calculate the optimal values of the variance terms
    '''
    
    n_dat_Xy = Xy.shape[0]
    Zmc = Zsamples[0]
    Wmc = Zsamples[1]    
    n_mc = Zmc.shape[0]    

    sse_l = np.zeros(Zmc[0,:,0].shape)
    sse_h = np.zeros(Y[0,:].shape)
    sig2_z = np.zeros(sse_l.shape)
    sig2_y = np.zeros(sse_h.shape)
    
    # For high-level X-Y data
    for n in range(n_dat_Xy):
        z_func = func_low( theta, Xy[n,:].reshape((1,-1)) )

        for i in range(n_mc):
            err = Zmc[i,:,n] - z_func
            sse_l += Wmc[i,n]* np.squeeze(err)**2
            y_func = func_top(theta, Xy[n,:], Zmc[i,:,n])
            err = Y[n,:] - y_func
            sse_h += Wmc[i,n]* np.squeeze(err)**2

    sig2_y = sse_h / n_dat_Xy
    
    # For low-level X-Z data
    d_z = len(Xz)

    # Data in Xz and Z are saved in lists, each item in the lists represent one
    #    Z-variable
    for i in range (d_z):
        Xtmp = Xz[i]
        Ztmp = Z[i]
        n_dat_Xz = Xtmp.shape[0]   
        for n in range(n_dat_Xz):
            z_func = func_low( theta, Xtmp[n,:].reshape((1,-1)) )
            err = Ztmp[n] - z_func[:,i]
            sse_l[i] += err*err

        sig2_z[i] = sse_l[i] / (n_dat_Xz+n_dat_Xy)

    return (sig2_y, sig2_z)

def pred(func_top, func_low, X, theta, sig2_y, sig2_z, V):
    ''' Function to make prediction (both mean and variance
    Args:
    theta - parameter (estimated)
    V - the covariance matrix of theta
    '''

    n_dat = X.shape[0]
    n_theta = theta.shape[0]
    
    # step 1: predict for lower-level model
    
    Z_mean = func_low(theta, X)
    grad_low = calc_grad_theta(func_low, theta, X)
    d_func_low = grad_low.shape[1] # grad_low in shape of [n_dat, d_func, n_paras]
    
    Z_cov = np.zeros((n_dat, d_func_low, d_func_low))
    for i in range(n_dat): 
        gd = np.mat(grad_low[i,:,:])
        #print gd
        Z = gd * V * np.matrix.transpose(gd) + np.diag(sig2_z)
        #print Z
        Z_cov[i,:,:] = np.array(Z)

        
    # step 2: predict for higher-level model
    
    Y_mean = func_top(theta, X, Z_mean)
    grad_high = calc_grad_theta_Z(func_top, theta, X, Z_mean)
    grad_high_theta = grad_high[0] # grad_high_theta in shape of [n_dat, d_func, n_paras]
    grad_high_Z = grad_high[1]     # grad_high_Z in shape of [n_dat, d_func, d_Z]
    d_func_high = grad_high_theta.shape[1] 

    Y_cov = np.zeros((n_dat, d_func_high, d_func_high))
    zero_mat = np.mat(np.zeros((V.shape[0], Z_cov.shape[1])))
    zero_mat_trans = np.matrix.transpose(zero_mat)
    for i in range(n_dat):
        Z = np.mat(Z_cov[i,:,:])
        block_mat = np.bmat([ [V, zero_mat], [zero_mat_trans, Z] ])
        gd = np.bmat( [ grad_high_theta[i,:,:], grad_high_Z[i,:,:] ] )
        Y = gd * block_mat * np.matrix.transpose(gd) + np.diag(sig2_y)
        #print Y
        Y_cov[i,:,:] = np.array(Y)

    return (Z_mean, Z_cov, Y_mean, Y_cov)   


###########################################################
def TransUnctnKF(func, paras_mean, paras_cov, X):
    ''' Function to transform the uncertainty, represented as a normal distribution (paras_mean, paras_cov)
    through a deterministic (func) to calculate the normal approximation of the output uncertainty
    X is the additional inputs that are needed for func
    '''

    y_mean = func(paras_mean, X)

    n_paras = len(paras_mean)
    
    grad = np.zeros( (n_paras, 1) )
    
    delta_paras = np.fabs(paras_mean) * 1e-3
    delta_paras[ delta_paras<1e-8 ] = 1e-8 # to avoid too small values

    # finite difference approximation of the gradient
    for i in range(n_paras):
        p1 = np.copy(paras_mean)
        p1[i] += delta_paras[i]
        t1 = func(p1, X)
        grad[i] = (t1-y_mean) / delta_paras[i]

    y_cov = np.transpose(grad).dot( paras_cov.dot(grad) )

    return (y_mean, y_cov)


