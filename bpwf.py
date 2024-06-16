"""
==========================
Bandpower window functions
==========================

"""

import numpy as np
from spectra import specind, specgen

class BPWF():
    """
    A BPWF object contains the bandpower window functions corresponding to the
    set of binned output spectra that can be calculated from a set of maps.
    Window functions are defined by the output spectrum (cross between two
    maps), the input spectrum (one of the six possible spectra from input
    TQU sky maps), and the ell bin.

    """
    
    def __init__(self, maplist, nbin, strict=False):
        """
        Creates a BPWF structure for the specified list of maps.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        nbin : int
            Number of ell bins, assumed to be the same for all spectra.
        strict : bool, optional
            If True, then the window_expv function will throw a KeyError if
            you request a window function which has not been defined (via the
            add_windowfn method). If False (default value), then expectation
            values will be zero for any undefined window function.

        """
        
        self.maplist = maplist
        self.nbin = nbin
        self.bpwf = {}
        self.strict = strict

    def add_windowfn(self, specin, m0, m1, windowfn, lmin=None, lmax=None):
        """
        Add a set of bandpower window functions to the BPWF object.

        Parameters
        ----------
        specin : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the input spectrum for the window functions
        m0 : int or str
            Identifier for map0 of the output spectrum. If this argument is an
            integer, it is interpreted as the index of the map in the object's
            maplist member. If this argument is a string, it is interpreted as
            the name of a map contained in the object's maplist member.
        m1 : int or str
            Identifier for map1 of the output spectrum. Interpretation of
            int vs str argument is described above.
        windowfn : array, shape (N,M)
            Two-dimensional array containing window functions for N ell bins,
            where N=self.nbin. The length of each window function, M, should
            be equal to lmax-lmin+1.
        lmin : int, optional
            Specify lmin value for the window functions. Default value is 0.
        lmax : int, optional
            Specify lmax value for the window functions. Default value is
            length of the window functions minus one.

        Returns
        -------
        None

        """

        # Check specin argument.
        if specin not in ['TT','EE','BB','TE','EB','TB']:
            raise ValueError('invalid specin argument')
        # Parse m0,m1 arguments.
        if type(m0) is str:
            m0 = [m.name for m in self.maplist].index(m0)
        assert m0 < len(self.maplist)
        if type(m1) is str:
            m1 = [m.name for m in self.maplist].index(m1)
        assert m1 < len(self.maplist)
        
        # Check that windowfn argument has the right shape.
        assert windowfn.shape[0] == self.nbin
        if lmin is None:
            lmin = 0
        if lmax is None:
            lmax = lmin + windowfn.shape[1]
        assert windowfn.shape[1] == (lmax - lmin)

        # Add window functions to the BPWF object.
        specout = specind(len(self.maplist), m0, m1)
        if specout not in self.bpwf.keys():
            self.bpwf[specout] = {}
        self.bpwf[specout][specin] = {'fn': windowfn,
                                      'lmin': lmin, 'lmax': lmax}

    def valid_windowfn(self, specin, m0, m1):
        """
        Check whether specified window function exists for this BPWF object.

        Parameters
        ----------
        specin : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying input spectrum type.
        m0 : int or str
            Identifier for map0 of the output spectrum. If this argument is an
            integer, it is interpreted as the index of the map in the object's
            maplist member. If this argument is a string, it is interpreted as
            the name of a map contained in the object's maplist member.
        m1 : int or str
            Identifier for map1 of the output spectrum. Interpretation of
            int vs str argument is described above.

        Returns
        -------
        valid : bool
            True if specified window function exists. False otherwise.

        """

        # Check specin argument.
        if specin not in ['TT','EE','BB','TE','EB','TB']:
            raise ValueError('invalid specin argument')
        # Parse m0,m1 arguments.
        if type(m0) is str:
            m0 = [m.name for m in self.maplist].index(m0)
        if type(m1) is str:
            m1 = [m.name for m in self.maplist].index(m1)
        if (m0 >= len(self.maplist)) or (m1 >= len(self.maplist)):
            raise ValueError('invalid map argument')
        # Check that we have the requested window functions.
        specout = specind(len(self.maplist), m0, m1)
        if specout not in self.bpwf.keys():
            return False
        if specin not in self.bpwf[specout].keys():
            return False
        # If we didn't fail previous two tests, then window fn exists.
        return True
        
    def window_expv(self, specin, m0, m1, fn):
        """
        Calculates the window-fn weighted sum for specified function.

        Parameters
        ----------
        specin : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the spectrum type of the supplied function
        m0 : int or str
            Identifier for map0 of the output spectrum. If this argument is an
            integer, it is interpreted as the index of the map in the object's
            maplist member. If this argument is a string, it is interpreted as
            the name of a map contained in the object's maplist member.
        m1 : int or str
            Identifier for map1 of the output spectrum. Interpretation of
            int vs str argument is described above.
        fn : function
            A function that takes an array of ell values as an input and
            returns an array of power spectrum values (for the type of
            spectrum given by specin argument).

        Returns
        -------
        expv : array, shape (N,)
            Array of bandpower expectation values for the N ell bins.

        """
        
        # Check that window function is defined.
        if not self.valid_windowfn(specin, m0, m1):
            if self.strict:
                raise KeyError('requested window function not defined')
            else:
                return np.zeros(self.nbin)

        # Parse m0,m1 arguments.
        if type(m0) is str:
            m0 = [m.name for m in self.maplist].index(m0)
        if type(m1) is str:
            m1 = [m.name for m in self.maplist].index(m1)
        specout = specind(len(self.maplist), m0, m1)
        
        # Calculate window-function weighted integrals.
        bpwf = self.bpwf[specout][specin]
        ell = np.arange(bpwf['lmin'], bpwf['lmax'])
        expv = np.zeros(self.nbin)
        for i in range(self.nbin):
            expv[i] = np.sum(fn(ell) * bpwf['fn'][i,:])
        return expv

    def ell_eff(self, specin, m0, m1):
        """
        Calculates effective ell values for bandpower window functions.

        Parameters
        ----------
        specin : {'TT', 'EE', 'BB', 'TE', 'EB', 'TB'}
            String specifying the input spectrum for window functions.
        m0 : int or str
            Identifier for map0 of the output spectrum. If this argument is an
            integer, it is interpreted as the index of the map in the object's
            maplist member. If this argument is a string, it is interpreted as
            the name of a map contained in the object's maplist member.
        m1 : int or str
            Identifier for map1 of the output spectrum. Interpretation of
            int vs str argument is described above.

        Returns
        -------
        lval : array, shape (N,)
            Array of effective ell values for the N bins.

        """

        return self.window_expv(specin, m0, m1, lambda x: x)

    def select(self, mapind=None, ellind=None):
        """
        Make a new BPWF object with selected maps and/or ell bins.

        Parameters
        ----------
        mapind : list, optional
            List of maps to keep for the new BPWF object. Maps can be specified
            either by their integer index in the existing maplist or by their
            names (as strings). Defaults to None, which means to *keep all
            maps*.
        ellind : list, optional
            List of ell bins to keep for the new BPWF object. Ell bin are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        bpwf_new : BPWF
            New BPWF object with selected maps and ell bins only.

        """

        # Process mapind argument.
        if mapind is None:
            mapind = range(len(self.maplist))
        for (i,val) in enumerate(mapind):
            if type(val) == str:
                mapind[i] = [m.name for m in maplist].index(val)
        # Process ellind argument.
        if ellind is None:
            ellind = range(self.nbin)

        # Create new BPWF object.
        maplist_new = [self.maplist[i] for i in mapind]
        nbin_new = len(ellind)
        bpwf_new = BPWF(maplist_new, nbin_new, strict=self.strict)

        # Copy window functions to new object.
        for (i, m0, m1) in specgen(len(maplist_new)):
            # Find the index of this spectra in old BPWF object.
            i0 = specind(len(self.maplist), mapind[m0], mapind[m1])
            # Copy BPWF
            bpwf_new.bpwf[i] = self.bpwf[i0]
        # Done
        return bpwf_new
