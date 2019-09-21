import numpy as np
import itertools

#====================================================================================================
# Adapted from "interpn_linear" from python 'interpolation' package
#====================================================================================================
class RegularGridInterpolator(object):
    """
    Interpolation on a regular grid in arbitrary dimensions

    The data must be defined on a regular grid; the grid spacing however may be
    uneven.  Linear and nearest-neighbour interpolation are supported. After
    setting up the interpolator object, the interpolation method (*linear* or
    *nearest*) may be chosen at each evaluation.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    method : str, optional
        The method of interpolation to perform. Supported are "linear" and
        "nearest". This parameter will become the default for the object's
        ``__call__`` method. Default is "linear".

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.

    Methods
    -------
    __call__

    Notes
    -----
    Contrary to LinearNDInterpolator and NearestNDInterpolator, this class
    avoids expensive triangulation of the input data by taking advantage of the
    regular grid structure.

    If any of `points` have a dimension of size 1, linear interpolation will
    return an array of `nan` values. Nearest-neighbor interpolation will work
    as usual in this case.

    .. versionadded:: 0.14


    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan):
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)
        self.method = method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        for i, p in enumerate(points):
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == len(p):
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values

    def __call__(self, xi, method=None):
        """
        Interpolation at coordinates

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        method : str
            The method of interpolation to perform. Supported are "linear" and
            "nearest".

        """
        method = self.method if method is None else method
        if method not in ["linear", "nearest"]:
            raise ValueError("Method '%s' is not defined" % method)

        ndim = len(self.grid)
        #xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterpolator has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)

        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)

        if method == "linear":
            result, deriv1, deriv2 = self._evaluate_linear(indices,
                                           norm_distances,
                                           out_of_bounds)
        elif method == "nearest":
            result = self._evaluate_nearest(indices,
                                            norm_distances,
                                            out_of_bounds)
        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:]), deriv1.reshape(xi_shape[:-1] + self.values.shape[ndim:]), deriv2.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_linear(self, indices, norm_distances, out_of_bounds):
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = itertools.product(*[[i, i + 1] for i in indices])
        values = 0.
        values_x = 0.
        values_y = 0.
        #coeff_matrix = np.zeros(((self.values.size),(self.values.size)))
        #coeff_matrix_full = np.zeros((2, (self.values.size), (self.values.size)))
        #index = np.asarray(range(0,(self.values.size)))
        for edge_indices in edges:
            weight = 1.
            weight_x = 1.
            weight_y = 1.
            count = 0
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
                if count==0:
                    weight_x *= np.where(ei == i, -1, 1)
                    weight_y *= np.where(ei == i, 1 - yi, yi)
                else:
                    weight_x *= np.where(ei == i, 1 - yi, yi)
                    weight_y *= np.where(ei == i, -1, 1)
                #weight2 *= np.where(ei == i, -0.5, 0.5)
                count += 1
            #weight_indices = edge_indices[0]*65 + edge_indices[1]
            #coeff_indices = (index,weight_indices);
            #coeff_matrix[coeff_indices] += weight[vslice]
            #coeff_matrix_full[0,:,:] = coeff_matrix

            values += np.asarray(self.values[edge_indices]) * weight[vslice]
            values_x += np.asarray(self.values[edge_indices]) * weight_x[vslice]
            values_y += np.asarray(self.values[edge_indices]) * weight_y[vslice]
        return values, values_x, values_y# coeff_matrix_full

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds):
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            idx_res.append(np.where(yi <= .5, i, i + 1))
        return self.values[idx_res]

    def _find_indices(self, xi):
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, norm_distances, out_of_bounds

#====================================================================================================

def interpn_linear(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular grids.
    Modified from interpn (scipy.interpolate package)

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    xi : ndarray of shape (..., ndim)
        The coordinates to sample the gridded data at

    bounds_error : bool, optional
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.

    fill_value : number, optional
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated.  Extrapolation is not supported by method
        "splinef2d".

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        Interpolated values at input coordinates.


    """
    # sanity check 'method' kwarg

    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    ndim = values.ndim
    # sanity check consistency of input dimensions
    if len(points) > ndim:
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), ndim))

    # sanity check input grid
    for i, p in enumerate(points):
        if not np.all(np.diff(p) > 0.):
            raise ValueError("The points in dimension %d must be strictly "
                             "ascending" % i)
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))
    grid = tuple([np.asarray(p) for p in points])

    # sanity check requested xi
    #xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("The requested sample points xi have dimension "
                         "%d, but this RegularGridInterpolator has "
                         "dimension %d" % (xi.shape[1], len(grid)))

    for i, p in enumerate(xi.T):
        if bounds_error and not np.logical_and(np.all(grid[i][0] <= p),
                                               np.all(p <= grid[i][-1])):
            raise ValueError("One of the requested xi is out of bounds "
                             "in dimension %d" % i)

    # perform interpolation
    if method == "linear":
        interp = RegularGridInterpolator(points, values, method="linear",
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        return interp(xi)

