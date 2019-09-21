from scipy import interpolate
from cost_function import *
from scipy.sparse import csr_matrix
import scipy.optimize as op


################################################################################
#
# Automatic registration
#
###############################################################################


def registration(u,v,x,y,I,c1=0.1,c2=1,c3=1,eps1=10**-5,eps2=10**-5):
    ny = len(y)
    nx = len(x)

    for i in range(1,I+1):
        print('Morphing step {}'.format(i))
        mi = 2 ** i + 1

        # ==============================================================================================================
        # 1. Smoothing
        # ==============================================================================================================
        vs = smooth(v, y, x, i)
        us = smooth(u, y, x, i)


        # ==============================================================================================================
        # 2. Initialization
        # ==============================================================================================================
        if i==1:
            # Create initial (3 by 3) regular grid
            yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
            xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)

            # Initialize variables
            mxT, myT = np.meshgrid(x[xc], y[yc], indexing='ij')
            grid = np.zeros((2 ** i + 1) ** 2 * 2 )
            grid[0:(2 ** i + 1) ** 2] = mxT.reshape(-1)
            grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2] = myT.reshape(-1)

        else:
            # New undistorted finer grid
            yc_new = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
            xc_new = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)

            # Interpolate T from coarser grid into the new grid
            xxc, ttc = np.meshgrid(x[xc_new], y[yc_new], indexing='ij')
            Ty = interpolate.interpn((x[xc], y[yc]), myT, np.array([xxc.reshape(-1), ttc.reshape(-1)]).T, method='linear')
            Tx = interpolate.interpn((x[xc], y[yc]), mxT, np.array([xxc.reshape(-1), ttc.reshape(-1)]).T, method='linear')

            # Initialize variables
            myT = Ty.reshape((mi, mi))
            mxT = Tx.reshape((mi, mi))
            grid = np.zeros((2 ** i + 1) ** 2 * 2)
            grid[0:(2 ** i + 1) ** 2] = mxT.reshape(-1)
            grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2] = myT.reshape(-1)

            # Reset grid
            xc = xc_new
            yc = yc_new


        # ==============================================================================================================
        # 3. Optimization
        # ==============================================================================================================

        # ==============================================================================================================
        # 3.1. We pre-compute the elements needed for the derivative of the cost function
        # that only depends on the original grid and i (i.e. independant from the mapping T)

        # Pre-computation of derivative elements non-dependant of the distortion
        dxdT, dydT = dXdT(y, x, i)
        dxdT = csr_matrix(dxdT)
        dydT = csr_matrix(dydT)

        # Pre-computation of the (spatial) derivation matrices
        Ax = np.diag(-1 * np.ones(mi * (mi - 1)), -mi) + np.diag(np.ones(mi * (mi - 1)), mi)
        Ax[0:mi, 0:mi] = -1 * np.eye(mi)
        Ax[mi * (mi - 1):mi ** 2, mi * (mi - 1):mi ** 2] = 1 * np.eye(mi)
        Ax[0:mi, :] = 1 / (xc[1] - xc[0]) * Ax[0:mi, :]
        Ax[mi * (mi - 1):mi ** 2, :] = 1 / (xc[1] - xc[0]) * Ax[mi * (mi - 1):mi ** 2, :]
        Ax[mi:mi * (mi - 1), :] = 1 / (xc[2] - xc[0]) * Ax[mi:mi * (mi - 1), :]
        ay = np.diag(-1 * np.ones(mi - 1), -1) + np.diag(np.ones(mi - 1), 1)
        ay[0, 0] = -1
        ay[mi - 1, mi - 1] = 1
        ay[0, :] = 1 / (yc[1] - yc[0]) * ay[0, :]
        ay[mi - 1, :] = 1 / (yc[1] - yc[0]) * ay[mi - 1, :]
        ay[1:mi - 1, :] = 1 / (yc[2] - yc[0]) * ay[1:mi - 1, :]
        temp = np.multiply.outer(np.eye(mi), ay)
        Ay = np.swapaxes(temp, 1, 2).reshape((mi ** 2, mi ** 2))
        Ax = csr_matrix(Ax)
        Ay = csr_matrix(Ay)

        # ==============================================================================================================
        # 3.2. Define bounds for minimization problem
        bnds = ()
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        for k in range(0, mi ** 2):
            bnds = bnds + ((xmin, xmax),)
        for k in range(mi ** 2, 2 * mi ** 2):
            bnds = bnds + ((ymin, ymax),)

        # ==============================================================================================================
        # 3.3. Minimization

        # Initialize barrier coefficient (b) and stopping criteria for barrier method
        b = 1*np.ones(4*(mi - 1)**2)
        crit1 = 1
        crit2 = 1

        # Iteration for barrier method
        while (crit1>eps1 or crit2>eps2) :
            # Minimization of the (penalized) cost function Jp
            tTo1 = op.minimize(Jp, grid, args=(b, us, vs, y, x, i,c1,c2,c3,dxdT, dydT, Ax, Ay), jac=True, method='L-BFGS-B', bounds=bnds,options={'maxiter':10000, 'maxfun': 100000})

            # Check if penalty criteria met
            if np.all(constr1(tTo1.x,i)>=0) and np.all(constr2(tTo1.x,i)>=0) and np.all(constr3(tTo1.x,i)>=0) and np.all(constr4(tTo1.x,i)>=0):
                crit1 = 0
                crit2 = 0
            else:
                crit1 = np.sqrt(np.sum((grid-tTo1.x)**2))
                crit2 = np.abs(Jp(grid, b, us, vs, y, x, i,c1,c2,c3, dxdT, dydT, Ax, Ay)[0]-Jp(tTo1.x,b, us, vs, y, x, i,c1,c2,c3, dxdT, dydT, Ax, Ay)[0])
            grid = tTo1.x
            b = 10*b

        # Update grid
        myT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
        mxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))

    # Return mapping
    return mxT, myT



