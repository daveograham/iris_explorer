import numpy as np

### TOOLS FOR SPECTRAL FITTING ###

def ngen_gauss(wav, *P):
    '''Takes a list [I_1, cen_1, sigma_1, I_N, cen_N, sigma_N..] or a numpy array of dims (Ngaussian, 3) with columns I, centroid, sigma'''
    P = np.asarray(P)

    if P.size % 3.0 != 0:
        print('Parameter number mismatch - expects array or list of I/centroid/sigma per line')
        return -1
    else:
        nprofiles = int(P.size/3.0)

        if P.ndim < 2:
            P = P.reshape((nprofiles,3))

        gauss_array = P[:,0][:,None] * np.exp(-(wav[None,:] - P[:,1][:,None])**2.0/(2.0 * (P[:,2][:,None]**2.0)))
        gauss_total = gauss_array.sum(axis=0)
        
        return gauss_total

def gaussian_bounds(P, ilow=False):
    '''Set default parameter boundaries from a list of starting parameters'''
    P = np.asarray(P)

    if P.size % 3.0 != 0:
        print('Parameter number mismatch - expects array or list of I/centroid/sigma per line')
        return -1
    else:
        ngauss = int(P.size/3.0)
        #if a list reshape into an array Ngauss X 3
        if P.ndim < 2:
            P = P.reshape((ngauss,3))

        def_lowI = 0.0

        deltaI = 0.3
        deltacen = 0.25
        lowsigma = 0.025
        highsigma = 2.0

        lowbound = []
        highbound = []

        for i in range(ngauss):
            #I
            if ilow is not False:
                lowbound.append(P[i,0]*(1-deltaI))
            else:
                lowbound.append(0.0)
            highbound.append(P[i,0]*(1+deltaI))
            #cen
            lowbound.append(P[i,1]-deltacen)
            highbound.append(P[i,1]+deltacen)
            #sigma
            lowbound.append(lowsigma)
            highbound.append(highsigma)

        return tuple((lowbound, highbound))