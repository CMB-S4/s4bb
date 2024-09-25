"""
===========================
Bandpower covariance matrix
===========================

Bandpower covariance matrix can be defined as a simple static matrix or a more
complicated object that can calculate bandpower covariance while varying noise
and/or signal levels. This implemented with the BpCov base class and derived
classes.

All bandpower covariance classes should include the following instance
variables and methods:
* maplist  : instance variable containing a list of MapDef objects
* nbin     : instance variable specifying the number of ell bins
* nmap()   : method that returns the number of maps, i.e. len(maplist)
* nspec()  : method that returns the number of spectra, i.e. nmap*(nmap+1)/2
* get()    : method that returns the bandpower covariance matrix
* select() : returns a new covariance matrix object defined for a subset of
             maps and/or ell bins

The bandpower covariance matrix returned by the get() function is organized
into blocks by ell bin. The size of each block is Nspec x Nspec, where
Nspec = Nmap * (Nmap + 1) / 2. Within a block, the spectra follow vecp
ordering, as discussed in spectra.py (which also contains helper functions).

"""

import numpy as np
from util import specind, specgen

class BpCov():
    """
    A BpCov object contains the bandpower covariance matrix for the full set
    of auto and cross spectra derived from a set of maps.

    This class can be used for a static, user-supplied bandpower covariance
    matrix or as a base class for more complex bpcm constructions.

    """
    
    def __init__(self, maplist, nbin, matrix=None):
        """
        Creates a BpCov object for specified list of maps and number of ell
        bins.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        nbin : int
            Number of ell bins, assumed to be the same for all spectra.
        matrix : array, shape (N,N), optional
            Array specifying a bandpower covariance matrix. This could be
            calculated analytically or derived from sims. The shape of this
            array must be (N,N), where N = nbin * nmap * (nmap + 1) / 2.

        """
        
        self.maplist = maplist
        self.nbin = nbin
        self.matrix = {}
        if matrix is not None:
            self.set(matrix)

    def nmap(self):
        """Returns the number of maps defined for this object."""
        
        return len(self.maplist)

    def nspec(self):
        """Returns the number of spectra defined for this object."""
        
        nmap = self.nmap()
        return nmap * (nmap + 1) // 2

    def set(self, matrix):
        """
        Assigns a (static) bandpower covariance matrix.

        Parameters
        ----------
        matrix : array, shape (N,N), optional
            Array specifying a bandpower covariance matrix. This could be
            calculated analytically or derived from sims. The shape of this
            array must be (N,N), where N = nbin * nmap * (nmap + 1) / 2.

        Returns
        -------
        None

        """
        
        # Check that matrix has the right size
        assert (self.nspec() * self.nbin == matrix.shape[0])
        assert (self.nspec() * self.nbin == matrix.shape[1])
        self.matrix['total'] = matrix
            
    def get(self, noffdiag=None):
        """
        Returns an array containing bandpower covariance matrix.

        Parameters
        ----------
        noffdiag : int, optional
            If set to a non-negative integer, then this parameter defines the
            maximum range in ell bins that is retained for offdiagonal
            bandpower covariance. For example, if noffdiag=1 then we keep the
            covariances between bin i and bins i-1, i, and i+1, but zero out
            the covariances with any other bins. Default value is None, which
            means to keep covariances between *all* ell bins.

        Returns
        -------
        M : array, shape (N,N)
            Bandpower covariance matrix. The shape of this array is (N,N),
            where N = nbin * nmap * (nmap + 1) / 2. If the noffdiag argument
            is set, then some offdiagonal blocks of this matrix may be set to
            zero.

        """

        # Returns the matrix with requested masking of off-diagonal blocks
        return self.matrix['total'] * self.mask_ell(noffdiag=noffdiag)

    def mask_ell(self, noffdiag=None):
        """
        Returns an array of ones with some offdiagonal blocks (optionally) set
        to zero.

        Parameters
        ----------
        noffdiag : int, optional
            If set to a non-negative integer, then this parameter defines the
            maximum range in ell bins that is retained for offdiagonal
            bandpower covariance. For example, if noffdiag=1 then we keep the
            covariances between bin i and bins i-1, i, and i+1, but zero out
            the covariances with any other bins. Default value is None, which
            means to keep covariances between *all* ell bins.

        Returns
        -------
        M : array, shape (N,N)
            Mask array with shape (N,N), where N = nbin * nmap * (nmap + 1) / 2.
            This array consists of blocks that are set to one if they are
            within noffdiag of the diagonal, or zero otherwise.
        
        """
        
        N = self.nspec() * self.nbin
        # By default, don't mask anything.
        if noffdiag is None:
            return np.ones(shape=(N,N))
        # If noffdiag is negative, set it to zero.
        if noffdiag < 0:
            noffdiag = 0
        # Otherwise, construct mask that is set to one for blocks that are
        # within noffdiag of the diagonal, zero otherwise.
        mask = np.zeros(shape=(N,N))
        for i in range(self.nbin):
            x0 = i * self.nspec()
            x1 = (i + 1) * self.nspec()
            y0 = max(0, i - noffdiag) * self.nspec()
            y1 = min(self.nbin, i + noffdiag + 1) * self.nspec()
            mask[x0:x1,y0:y1] = 1.0
        return mask
    
    def select(self, maplist=None, ellind=None):
        """
        Make a new BpCov object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new BpCov object. Defaults to None,
            which means that the new BpCov object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new BpCov object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        bpcov_new : BpCov
            New BpCov object with updated maps and ell bins.        

        """

        # Process maplist argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
        # Process ellind argument.
        if ellind is None:
            ellind = range(self.nbin)

        # Create new BpCov object
        bpcov_new = BpCov(maplist, len(ellind))
            
        # Determine mapping from old to new bandpower ordering.
        ind0 = np.zeros(bpcov_new.nspec())
        for (i,m0,m1) in specgen(bpcov_new.nmap()):
            i0 = self.maplist.index(bpcov_new.maplist[m0])
            i1 = self.maplist.index(bpcov_new.maplist[m1])
            ind0[i] = specind(self.nmap(), i0, i1)
        # Expand over selected ell bins.
        ind1 = np.zeros(bpcov_new.nspec() * bpcov_new.nbin, dtype=int)
        for (i,x) in enumerate(ellind):
            ind1[i*bpcov_new.nspec():(i+1)*bpcov_new.nspec()] = ind0 + x*self.nspec()

        # Apply this selection to all keys in the matrix dict, so that this
        # function will still work for more complicated derived classes.
        for key in self.matrix.keys():
            bpcov_new.matrix[key] = self.matrix[key][np.ix_(ind1,ind1)]

        # Done
        return bpcov_new
