import numpy as np

### VARIOUS TOOLS FOR SPECTRAL FITTING ###
def hfit(value, omin, omax, nmin, nmax):
    '''houdini style fit function'''
    norm = np.clip((value - omin) / (omax - omin), 0.0, 1.0)
    return (1-norm)*nmin + norm*nmax


def single_gauss(wav, I, wav0, sigma):
    '''I[0], wav0[1], sigma[2]'''
    ygauss = I * np.exp(-(wav - wav0)**2.0/(2.0 * (sigma**2.0)))
    return ygauss


def ngen_gauss_old(wav, *P):
    '''Takes parameters P in form: I[0], wav0[1], sigma[2], I[1]....'''

    if len(P)%3 == 0:
        nprofiles = int(np.floor(len(P)/3))
        ygauss = 0.0

        for n in range(nprofiles):
            ygauss += P[3*n] * np.exp(-(wav - P[3*n+1])**2.0/(2.0 * (P[3*n+2]**2.0)))

        return ygauss
    else:
        print('Parameter number mismatch - expects sets of 3 - I/centroid/sigma')
        return 0
    
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
    

def gaussian_setup(wav, spec, ngauss=1, **kwargs):
    '''Set default starting parameters and boundaries. Takes: wavelength base, example spectrum,
      number of gaussians (optional), optional starting parameters params=[I,cen,sigma] '''

    #percentage range for limits
    deltaI = 0.3
    deltacen = 0.25
    deltasig = 0.5
    #sigma width limits
    lowsigma = 0.025
    highsigma = 2.0

    if 'params' in kwargs.keys():
        P = np.asarray(kwargs['params'])
        #if P.size % 3.0 != 0:
        #    print('Parameter number mismatch - expects array or list of I/centroid/sigma per line')
        #    return -1
        #else:

        ngauss = int(P.size/3.0)
        #if a list reshape into an array Ngauss X 3
        if P.ndim < 2:
            P = P.reshape((ngauss,3))
    else:
        #default starting I
        imax = np.max(spec)
        i_start = imax * (1.0 - (0.1 * float(ngauss)))
        #default starting centroids
        wx = wav.size
        if ngauss % 2 == 0:
            ldelta = int(wx / (ngauss+1))
            lx = np.arange(0,wx,ldelta)
            if ngauss == 2:
                lx = lx[1:]
            else:
                lx = lx[1:-1]
        else:
            ldelta = int(wx / (ngauss+1))
            lx = np.arange(0,wx,ldelta)
            lx = lx[1:-1]
        cen_start = wav[lx]
        #default starting width
        omin = 0
        omax = 5000.0

        norm = np.clip((i_start - omin) / (omax - omin), 0.0, 1.0)
        w_start = (1-norm) * lowsigma + norm * highsigma

        P = np.zeros((ngauss,3))
        #SET STARTING PARAMETERS
        for i in range(ngauss):
            P[i,0] = i_start
            P[i,1] = cen_start[i]
            P[i,2] = w_start
        #========================================================
    
    lowbound = []
    highbound = []

    for i in range(ngauss):
        #I
        lowbound.append(P[i,0]*(1-deltaI))
        highbound.append(P[i,0]*(1+deltaI))
        #cen
        lowbound.append(P[i,1]-deltacen)
        highbound.append(P[i,1]+deltacen)
        #sigma
        lowbound.append(np.clip(P[i,2]*(1-deltasig), lowsigma, highsigma))
        highbound.append(np.clip(P[i,2]*(1+deltasig), lowsigma, highsigma))

    return P, tuple((lowbound, highbound))



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


def gaussian_bounds_old(pstart):
    '''Set default parameter boundaries from a list of parameters'''
    ngauss = np.floor(int(len(pstart)/3.0))

    deltaI = 0.4
    deltacen = 0.3
    lowsigma = 0.025
    highsigma = 2.0

    lowbound = []
    highbound = []
    for i in range(ngauss):
        #I
        lowbound.append(pstart[i*3]*(1-deltaI))
        highbound.append(pstart[i*3]*(1+deltaI))
        #cen
        lowbound.append(pstart[i*3+1]-deltacen)
        highbound.append(pstart[i*3+1]+deltacen)
        #sigma
        lowbound.append(lowsigma)
        highbound.append(highsigma)

    return tuple((lowbound, highbound))

def start_params(spec, ngauss, *P):
    '''set start parameters. Requires number gaussians. Optional parameters in form I_1.. I_N, C_1.. C_N, W_1.. W_N.
    Defaults are set for missing int centroid or width'''
    P1 = np.max(spec)
    return P1