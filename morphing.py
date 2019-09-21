import numpy as np
from scipy import interpolate

################################################################################
#
# Morphing and related transformation
#
###############################################################################


def morphing(u, v, t, x, ttT, xxT, i, la):
    # Morph field u according to mapping T (ttT,xxT) and lambda (la)
    # u: field to morph
    # v: target field
    # t,x : coordinate of pixel grid
    # ttT, xxT: mapping T
    # i: number of the morphing grid on which T is given
    # lambda: morphing coefficient

    nt = len(t)
    nx = len(x)
    tc = range(0, nt, int((nt - 1) / 2 ** i))
    xc = range(0, nx, int((nx - 1) / 2 ** i))
    xx, tt = np.meshgrid(x, t, indexing='ij')
    xxc, ttc = np.meshgrid(xc, tc, indexing='ij')

    # Transform coordinate
    Tt = interpolate.interpn((xc, tc), ttT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear',
                             bounds_error=False, fill_value=None)
    tt_prime = Tt.reshape((nx, nt))
    Tx = interpolate.interpn((xc, tc), xxT, np.array([xx.reshape(-1), tt.reshape(-1)]).T, method='linear',
                             bounds_error=False, fill_value=None)
    xx_prime = Tx.reshape((nx, nt))

    # v inverse transform
    points = np.array([xxT.reshape(-1), ttT.reshape(-1)]).T
    values = xxc.reshape(-1)
    xxT_inv = interpolate.griddata(points, values, (xx, tt), method='linear')
    ttT_inv = interpolate.griddata(points, ttc.reshape(-1), (xx, tt), method='linear')
    v_inv = interpolate.interpn((x, t), v, np.array([xxT_inv.reshape(-1), ttT_inv.reshape(-1)]).T, method='linear',
                                bounds_error=False, fill_value=0)
    v_inv = v_inv.reshape((nx, nt))
    v_inv[np.isnan(v_inv)] = 0

    # morphed coordinates
    xxl = xx + la * (xx_prime - xx)
    ttl = tt + la * (tt_prime - tt)

    # Morphed signal
    u_morph = interpolate.interpn((x, t), (1.0 - la) * u + la * v_inv, np.array([xxl.reshape(-1), ttl.reshape(-1)]).T,
                              method='linear', bounds_error=False, fill_value=0)

    return u_morph.reshape(nx, nt)



def distort_grid(t,x,ttT, xxT,i, la):
    nt = len(t)
    nx = len(x)
    xx, tt = np.meshgrid(x, t, indexing='ij')
    tc = range(0, nt, int((nt - 1) / 2 ** i))
    xc = range(0, nx, int((nx - 1) / 2 ** i))
    xxc, ttc = np.meshgrid(xc, tc, indexing='ij')

    # Inverse transform coordinates
    points = np.array([(xxc + la * (xxT - xxc)).reshape(-1), (ttc + la*(ttT-ttc)).reshape(-1)]).T
    xx_l = interpolate.griddata(points, xxc.reshape(-1), (xx, tt), method='linear')
    tt_l = interpolate.griddata(points, ttc.reshape(-1), (xx, tt), method='linear')

    return xx_l.reshape(nx,nt),tt_l.reshape(nx,nt)
