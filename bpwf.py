"""
==========================
Bandpower window functions
==========================

Bandpower window functions describe the mapping of power from true sky signals
to measured bandpowers, including the effects of beams, filtering, partial sky
coverage, map apodization, etc.

We use a very general treatment of bandpower window functions, so that any
measured bandpower can have up to six window functions describing the transfer
functions from true sky TT, EE, BB, TE, EB, and TB. In practice, many of these
functions will be zero, e.g. TT->BB window function is zero in the absence of
temperature-to-polarization leakage.

Some conventions followed here:
* Expectation values are calculated as the theory spectrum multiplied by the
  window function and summed over ell values. The range of the summation is
  set by the ell values for which the window function is defined. It is not
  necessary to define the window function at Delta-ell=1 intervals, but no
  Delta-ell factor is included in the sum, so this might have unanticipated
  effects.
* We do not enforce any normalization for the bandpower window functions, so
  window functions with the same shape but different amplitudes will yield
  different expectation values. The purpose of this choice is so that the
  overall signal suppression factor can be encoded as the normalization.
* We assume that all binned output spectra share a common set of ell bins.
  This assumption could be relaxed if someone has a good reason.

"""

import numpy as np
from .util import specind, specgen

class BPWF():
    """
    A BPWF object contains the bandpower window functions corresponding to the
    set of binned output spectra that can be calculated from a set of maps.

    The purpose of this object is to calculate bandpower expectation values
    from unbinned theory spectra.

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
            If True, then the expv function will throw a KeyError if you
            request a window function which has not been defined (via the
            add_windowfn method). If False (default value), then expectation
            values will be zero for any undefined window function.

        """
        
        self.maplist = maplist
        self.nmap = len(maplist)
        self.nbin = nbin
        self.bpwf = {}
        self.strict = strict

    @classmethod
    def tophat(cls, maplist, bin_edges, lmin=None, lmax=None, strict=False):
        """
        Creates BPWF object with tophat window functions defined by ell bin
        edges.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define possible output spectra.
        bin_edges : list of int
            A list of bin edges for the tophat ell bins. The number of ell bins
            will be one less than the number of entries in this list.
        lmin : int, optional
            The minimum ell value over which to define window functions.
            Default behavior is to use the lower edge of the first ell bin.
        lmax : int, optional
            The maximum ell value over which to define window functions.
            Default behavior is to use the upper edge of the last ell bin.
        strict : bool, optional
            Defines behavior for the new BPWF object. Default = False.

        Returns
        -------
        bpwf : BPWF object
            New object containing tophat window functions for the specified
            spectra.

        """
        
        # Make tophat window functions
        if lmin is None:
            lmin = bin_edges[0]
        if lmax is None:
            lmax = bin_edges[-1]
        nbin = len(bin_edges) - 1
        fn = np.zeros(shape=(nbin, lmax - lmin))
        for i in range(len(bin_edges) - 1):
            i0 = bin_edges[i] - lmin
            i1 = bin_edges[i+1] - lmin
            fn[i,i0:i1] = 1 / (i1 - i0)
        # Create BPWF object
        bpwf = cls(maplist, nbin)
        for (i, m0, m1) in specgen(len(maplist)):
            specin = maplist[m0].field + maplist[m1].field
            if specin == 'ET': specin = 'TE'
            if specin == 'BE': specin = 'EB'
            if specin == 'BT': specin = 'TB'
            bpwf.add_windowfn(specin, m0, m1, fn, lmin, lmax)
        # Done
        return bpwf
        
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
        
    def expv(self, specin, m0, m1, fn):
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
        expvals = np.zeros(self.nbin)
        for i in range(self.nbin):
            expvals[i] = np.sum(fn(ell) * bpwf['fn'][i,:])
        return expvals

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

        return self.expv(specin, m0, m1, lambda x: x)

    def select(self, maplist=None, ellind=None):
        """
        Make a new BPWF object with selected maps and/or ell bins.

        This function can be used to downselect maps or ell bins from a BPWF
        object. Window functions will be copied over from the existing BPWF
        object to the new BPWF object.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new BPWF object. Defaults to None,
            which means that the new BPWF object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new BPWF object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        bpwf_new : BPWF
            New BPWF object with updated maps and ell bins.

        """

        # Process mapind argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
        # Find mapping between new and old map lists.
        # Newly added maps are marked with None.
        mapind = []
        for m in maplist:
            mapind.append(self.maplist.index(m))

        # Create new BPWF object.
        if ellind is not None:
            nbin = len(ellind)
        else:
            nbin = self.nbin
        bpwf_new = BPWF(maplist, nbin, strict=self.strict)

        # Copy window functions to new object.
        for (i, m0, m1) in specgen(len(maplist)):
            # Find the index of this spectra in old BPWF object.
            i0 = specind(len(self.maplist), mapind[m0], mapind[m1])
            # Copy BPWF -- have to do this manually to avoid duplicate
            # references to window function np arrays.
            bpwf_new.bpwf[i] = {}
            for spec in ['TT','EE','BB','TE','EB','TB']:
                if spec in self.bpwf[i0].keys():
                    bpwf_new.bpwf[i][spec] = {'fn': self.bpwf[i0][spec]['fn'].copy(),
                                              'lmin': self.bpwf[i0][spec]['lmin'],
                                              'lmax': self.bpwf[i0][spec]['lmax']}
            # If ellind argument is specified, keep only those ell bins.
            if ellind is not None:
                for key in bpwf_new.bpwf[i].keys():
                    bpwf_new.bpwf[i][key]['fn'] = (
                        bpwf_new.bpwf[i][key]['fn'][ellind,:])
        # Done
        return bpwf_new
