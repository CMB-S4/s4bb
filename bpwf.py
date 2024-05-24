"""
==========================
Bandpower window functions
==========================

"""

import numpy as np

class BPWF():
    def __init__(self, maplist, nbin, strict=False):
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
            lmax = windowfn.shape[0] - 1
        assert windowfn.shape[1] == (lmax - lmin + 1)

        # Add window functions to the BPWF object.
        specout = (min(m0,m1), max(m0,m1))
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
        specout = (min(m0, m1), max(m0, m1))
        if specout not in self.bpwf.keys():
            return False
        if specin not in self.bpwf[specout].keys():
            return False
        # If we didn't fail previous two tests, then window fn exists.
        return True
        
    def window_integral(self, specin, m0, m1, fn):
        """
        Calculates the window-fn weighted integral for specified function.

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
        specout = (min(m0, m1), max(m0, m1))
        
        # Calculate window-function weighted integrals.
        bpwf = self.bpwf[specout][specin]
        ell = np.arange(bpwf['lmin'], bpwf['lmax'] + 1)
        expv = np.zeros(self.nbin)
        for i in range(self.nbin):
            expv[i] = (np.trapz(fn(ell) * bpwf['fn'][i,:], x=ell) /
                       np.trapz(bpwf['fn'][i,:], x=ell))
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

        return self.window_integral(specin, m0, m1, lambda x: x)
