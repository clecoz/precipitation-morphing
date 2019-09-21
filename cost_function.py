import numpy as np
from scipy import interpolate
from interpolation import interpn_linear

################################################################################
#
# Cost function and related functions
#
###############################################################################

# ======================================================================================================================
# Smoothing function

def smooth(v,t,x,i):
    # Return smoothed signal
    v1 = np.zeros(v.shape)
    nx,nt = v.shape
    alpha = 0.05 / (2 ** (i * 2) + 1)
    for j in np.arange(0, nt):
        j = int(j)
        tloc = t[j]
        kernel_t = np.exp(-((t - tloc) / nt) ** 2 / alpha) / sum(np.exp(-((t - t[int((nt - 1) / 2)]) / nt) ** 2 / alpha))
        for k in np.arange(0, nx):
            k = int(k)
            xloc = x[k]
            kernel_x = np.exp(-((x - xloc) / nx) ** 2 / alpha) / sum(np.exp(-((x - x[int((nx - 1) / 2)]) / nx) ** 2 / alpha))
            v1[k,j] = np.dot(np.dot(v,kernel_t),kernel_x)
    return v1


# ======================================================================================================================
# Warping functions

def mapped(u,y,x,yyT,xxT,i):
    # Return the warped signal
    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Transform coordinate
    Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')
    Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), yy.reshape(-1)]).T, method='linear')

    # Interpolated function
    uT = interpolate.interpn((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None)  # ,fill_value=1000)
    return uT.reshape(nx,ny)

def mapped_TAHMO(u,y,x,yyT,xxT,lat_sta,lon_sta,i):
    # Return the values of the warped signal at given coordinates (lat_sta,lon_sta)
    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)

    # Transform coordinate
    Tt = interpolate.interpn((x[xc], y[yc]), yyT, np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False ,fill_value=None)
    Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([lon_sta, lat_sta]).T, method='linear',bounds_error=False, fill_value=None)

    # Interpolated function
    uT = interpolate.interpn((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=None)  # ,fill_value=1000)
    return uT


def mapped_weight(u,y,x,ttT,xxT,i):
    # Same as 'mapped' function, but also return weight from interpolation
    # Used in the computation of the derivative of the cost function
    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xx, tt = np.meshgrid(x, y, indexing='ij')

    # Transform coordinate
    Tt = interpolate.interpn((x[xc], y[yc]), ttT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')
    Tx = interpolate.interpn((x[xc], y[yc]), xxT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear')

    # Interpolated function
    uT, uT_x, uT_y = interpn_linear((x, y), u, np.array([Tx, Tt]).T, method='linear',bounds_error=False, fill_value=0)  # ,fill_value=1000)

    return uT.reshape(nx,ny), uT_x, uT_y


# ======================================================================================================================

def dXdT(t, x, i):
    nt = len(t)
    nx = len(x)
    mi = 2**i+1
    dnx = int((nx - 1) / 2 ** i)
    dnt = int((nt - 1) / 2 ** i)
    x1 = x[:, np.newaxis]

    dxdT2 = np.zeros((nx * nt, 2 * mi * mi))
    dtdT2 = np.zeros((nx * nt, 2 * mi * mi))
    for j in range(1, mi - 1):
        for k in range(1, mi - 1):
            B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 > x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t > t[dnt * j])
            B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t > t[dnt * j])
            B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 > x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
            B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
            dxdT2[:, k * mi + j] = (B1 + B2 + B3 + B4).reshape(-1)

        k = 0
        B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t < t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, j] = (B1 + B3).reshape(-1)
        k = mi - 1
        B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t < t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, (mi - 1) * mi + j] = (B2 + B4).reshape(-1)

    j = 0
    for k in range(1, mi - 1):
        B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 < x1[dnx * k]) * ( x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
        dxdT2[:, k * mi] = (B1 + B2).reshape(-1)
    j = mi - 1
    for k in range(1, mi - 1):
        B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
        B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ( (x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 < x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
        dxdT2[:, k * mi + j] = (B3 + B4).reshape(-1)

    j = 0
    k = 0
    B1 = (x1[dnx * (k + 1)] - x1) * (t[dnt * (j + 1)] - t) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * ( t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
    dxdT2[:, k * mi + j] = (B1).reshape(-1)
    k = mi - 1
    B2 = (x1 - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * (j + 1)] - t[dnt * j])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * (j + 1)]) * (t >= t[dnt * j])
    dxdT2[:, k * mi + j] = (B2).reshape(-1)
    k = 0
    j = mi - 1
    B3 = (x1[dnx * (k + 1)] - x1) * (t - t[dnt * (j - 1)]) / ((x1[dnx * (k + 1)] - x1[dnx * k]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * (k + 1)]) * (x1 >= x1[dnx * k]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
    dxdT2[:, k * mi + j] = (B3).reshape(-1)
    k = mi - 1
    B4 = (x1 - x1[dnx * (k - 1)]) * (t - t[dnt * (j - 1)]) / ((x1[dnx * k] - x1[dnx * (k - 1)]) * (t[dnt * j] - t[dnt * (j - 1)])) * (x1 <= x1[dnx * k]) * (x1 >= x1[dnx * (k - 1)]) * (t <= t[dnt * j]) * (t >= t[dnt * (j - 1)])
    dxdT2[:, k * mi + j] = (B4).reshape(-1)

    dtdT2[:, mi * mi:2 * mi * mi] = dxdT2[:, 0:mi * mi]

    return dxdT2, dtdT2


# ======================================================================================================================
# Constraints functions

def constr1(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2**i+1,2**i+1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2**i+1,2**i+1))
    c1 = (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 0:mi - 1]) * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]) - (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,0:mi - 1]) * (xxT[1:mi,1:mi] - xxT[0:mi - 1,0:mi - 1])
    return c1.reshape(-1)

def constr2(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c2 = (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]) * (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]) - (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]) * (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1])
    return c2.reshape(-1)

def constr3(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c3 = (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]) * (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]) - (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]) * (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi])
    return c3.reshape(-1)

def constr4(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))
    c4 = (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]) * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]) - (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]) * (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi])
    return c4.reshape(-1)


#===========================================================
# Derivative of the constraint functions

def dc1dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc1 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[1:mi, 0:mi-1] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc1dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc1dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)

    dc1dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (xxT[1:mi, 0:mi - 1] - xxT[0:mi-1, 0:mi-1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 0:mi - 1] - xxT[0:mi-1, 0:mi-1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[1:mi,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc1dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] +=  (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)

    Jc1[:,0:mi**2] = dc1dx
    Jc1[:,mi**2:] = dc1dt
    return Jc1

def dc2dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc2 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc2dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[0:mi - 1, 1:mi] - ttT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc2dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi,1:mi] - ttT[0:mi - 1,0:mi - 1]).reshape(-1)

    dc2dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 1:mi] - xxT[0:mi - 1, 0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)
    dc2dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] +=  (xxT[0:mi - 1,1:mi] - xxT[0:mi - 1,0:mi - 1]).reshape(-1)

    Jc2[:,0:mi**2] = dc2dx
    Jc2[:,mi**2:] = dc2dt
    return Jc2

def dc3dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc3 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[1:mi, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[1:mi, 1:mi] = np.ones((mi - 1, mi - 1))

    dc3dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi, 1:mi] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)
    dc3dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[1:mi,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)

    dc3dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[1:mi, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]).reshape(-1)
    dc3dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] +=  (xxT[1:mi, 1:mi] - xxT[0:mi - 1,1:mi]).reshape(-1)

    Jc3[:,0:mi**2] = dc3dx
    Jc3[:,mi**2:] = dc3dt
    return Jc3

def dc4dX(grid,i):
    mi = 2 ** i + 1
    xxT = grid[0:(2 ** i + 1) ** 2].reshape((2 ** i + 1, 2 ** i + 1))
    ttT = grid[(2 ** i + 1) ** 2:(2 ** i + 1) ** 2 * 2].reshape((2 ** i + 1, 2 ** i + 1))

    Jc4 = np.zeros(((mi-1)**2,2*mi**2))

    b = np.zeros((mi,mi))
    b[0:mi - 1, 1:mi] = np.ones((mi-1,mi-1))
    c = np.zeros((mi, mi))
    c[1:mi, 0:mi - 1] = np.ones((mi - 1, mi - 1))
    d = np.zeros((mi, mi))
    d[0:mi - 1, 0:mi - 1] = np.ones((mi - 1, mi - 1))

    dc4dx = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (ttT[1:mi, 0:mi - 1] - ttT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += -1 * (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)
    dc4dx[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += (ttT[0:mi - 1,0:mi - 1] - ttT[0:mi - 1,1:mi]).reshape(-1)

    dc4dt = np.zeros(((mi - 1) ** 2, mi ** 2))
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), c.reshape(-1) == 1] += (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] += -1 * (xxT[0:mi - 1, 0:mi - 1] - xxT[0:mi - 1, 1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), d.reshape(-1) == 1] += -1 * (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi]).reshape(-1)
    dc4dt[np.asarray(range(0, (mi - 1) ** 2)), b.reshape(-1) == 1] +=  (xxT[1:mi,0:mi - 1] - xxT[0:mi - 1,1:mi]).reshape(-1)

    Jc4[:,0:mi**2] = dc4dx
    Jc4[:,mi**2:] = dc4dt
    return Jc4


# ======================================================================================================================
# Cost function

def Jp(grid,b,u,v,y,x,i,c1,c2,c3,dxdT,dydT,Ax,Ay,mask=None):
    # Penalized cost function Jp with derivative
    mi = 2**i + 1

    ny = len(y)
    nx = len(x)
    yc = np.linspace(0, ny - 1, (2 ** i + 1), dtype=int)
    xc = np.linspace(0, nx - 1, (2 ** i + 1), dtype=int)
    xxc, yyc = np.meshgrid(x[xc], y[yc], indexing='ij')
    yyT = grid[(mi)**2:(mi)**2*2].reshape((mi,mi))
    xxT = grid[0:(mi)**2].reshape((mi, mi))
    b1 = b[0 : (mi-1)**2]
    b2 = b[(mi-1)**2 : 2*(mi - 1)**2]
    b3 = b[2*(mi-1)**2 : 3*(mi - 1)**2]
    b4 = b[3*(mi-1)**2 :  4*(mi - 1)**2]

    v1 = v
    u1 = mapped(u, y, x, yyT, xxT, i)

    ydif = (yyT-yyc).reshape(-1)
    xdif = (xxT-xxc).reshape(-1)

    dT = np.zeros(4 * mi ** 2)
    dT[0:mi ** 2] = Ax @ xdif
    dT[mi ** 2:2 * mi ** 2] = Ay @ xdif
    dT[2 * mi ** 2:3 * mi ** 2] = Ax @ ydif
    dT[3 * mi ** 2:4 * mi ** 2] = Ay @ ydif

    divT = Ax @ xdif + Ay @ ydif

    # Cost
    if mask is None:
        Jo = np.sqrt(sum(sum((v1 - u1) ** 2)))
    else:
        Jo = np.sqrt(sum(sum(mask * (v1 - u1) ** 2)))

    Jb = c1 /mi * np.sqrt(np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) \
               + c2 /mi * np.sqrt(np.dot(dT.T, dT)) \
               + c3 / mi * np.sqrt(np.dot(divT.T, divT))
    cost = Jo + Jb + (np.dot(b1 * (constr1(grid, i) < 0), constr1(grid, i) ** 2) + np.dot(b2 * (constr2(grid, i) < 0),constr2(grid, i) ** 2)
                    + np.dot(b3 * (constr3(grid, i) < 0), constr3(grid, i) ** 2) + np.dot(b4 * (constr4(grid, i) < 0),constr4(grid, i) ** 2))


    # Derivative
    u_w, u_x, u_y = mapped_weight(u, y, x, yyT, xxT, i)
    dx = round(x[1]-x[0],2)
    dy = round(y[1]-y[0],2)

    if mask is None:
        jac = - ((sum(sum((v1 - u1) ** 2))) ** (-1 / 2)) * ((((v1 - u1)).reshape(-1) * (u_y/dy) @ dydT) + (((v1 - u1)).reshape(-1) * (u_x/dx) @ dxdT))
    else:
        jac = - ((sum(sum(mask*(v1 - u1) ** 2))) ** (-1 / 2)) * (((mask*(v1 - u1)).reshape(-1) * (u_y / dy) @ dydT) + ((mask*(v1 - u1)).reshape(-1) * (u_x / dx) @ dxdT))

    if (np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) == 0 or c1 == 0:
        jac = jac
    else:
        jac2 = np.zeros(len(grid))
        jac2[0:mi ** 2] = xdif
        jac2[mi ** 2:mi ** 2 * 2] = ydif
        jac = jac + c1 /mi  * ((np.dot(ydif, ydif) + np.dot(xdif.T, xdif)) ** (-1 / 2)) * jac2

    if (np.dot(dT.T, dT)) == 0 or c2 == 0:
        jac = jac
    else:
        jac3 = np.zeros(len(grid))
        jac3[0:mi ** 2] = Ax.T @ dT[0:mi ** 2] + Ay.T @ dT[mi ** 2:2 * mi ** 2]
        jac3[mi ** 2:mi ** 2 * 2] = Ax.T @ dT[2 * mi ** 2:3 * mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2]
        jac = jac + c2 /mi  * (np.dot(dT.T, dT) ** (-1 / 2)) * jac3

    if (np.dot(divT.T,divT)) == 0 or c3 == 0:
        jac = jac
    else:
        jac5 = np.zeros(len(grid))
        jac5[0:mi ** 2] = (Ax.T @ dT[0:mi ** 2] + Ax.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac5[mi ** 2:mi ** 2 * 2] =  (Ay.T @ dT[0:mi ** 2] + Ay.T @ dT[3 * mi ** 2:4 * mi ** 2])
        jac = jac + c3 / mi  * (np.dot(divT.T,divT) ** (-1 / 2)) * jac5


    jac4 = 2 * (b1 * (constr1(grid,i) < 0) * constr1(grid,i)) @ dc1dX(grid,i) + 2 * (b2 * (constr2(grid,i) < 0) * constr2(grid,i)) @ dc2dX(grid,i) + 2 * (b3 * (constr3(grid,i) < 0) * constr3(grid,i)) @ dc3dX(grid,i) + 2 * (b4 * (constr4(grid,i) < 0) * constr4(grid,i)) @ dc4dX(grid,i)

    jac = jac + jac4

    return cost, jac


# ======================================================================================================================
# Registration

