"""
=============
Power spectra
=============

"""

import numpy as np

def specind(nmap, m0, m1):
    """
    Calculates the index of the specified spectrum in vecp ordering.

    Parameters
    ----------
    nmap : int
        Number of maps in the cross-spectral analysis
    m0 : int
        Index of map0 in the map list
    m1 : int
        Index of map1 in the map list

    Returns
    -------
    specind : int
        Index of the map0 x map1 spectrum in the list of spectra

    Notes
    -----
    Return value doesn't depend on the ordering of i0 vs i1.

    """

    # Check arguments
    if (nmap < 0) or (m0 < 0) or (m1 < 0):
        raise ValueError('arguments must be non-negative')
    if (m0 >= nmap) or (m1 >= nmap):
        raise ValueError('arguments m0 and m1 must be less than nmap')
    # Sort the map indices
    a = min(m0, m1)
    b = max(m0, m1)
    # Calculate spectrum index in vecp ordering
    return (sum(range(nmap, nmap - (b - a), -1)) + a)

def mapind(nspec, s0):
    """
    Calculates the map indices for a particular spectrum.

    Parameters
    ----------
    nspec : int
        Number of spectra in the cross-spectral analysis
    s0 : int
        Index of the spectrum

    Returns
    -------
    m0 : int
        Index of map0
    m1 : int
        Index of map1

    """

    # Check arguments
    if (nspec < 0) or (s0 < 0):
        raise ValueError('arguments must be non-negative')
    if (s0 >= nspec):
        raise ValueError('argument i must be less than nspec')
    # Calculate map indices
    nmap = int(-0.5 + np.sqrt(0.25 + 2 * nspec))
    for i in range(nmap):
        if s0 >= (nmap - i):
            s0 = s0 - (nmap - i)
        else:
            m0 = s0
            m1 = s0 + i
            break
    return (m0, m1)

def specgen(nmap):
    """
    Generator function for iterating over spectra in vecp ordering.

    Parameters
    ----------
    nmap : int
        Number of maps to iterate over.

    Yields
    ------
    i : int
        Spectrum index, starts at 0 and increments by 1.
    m0 : int
        Map 1 index
    m1 : int
        Map 2 index

    Example
    -------
    >>> nmap = 3
    >>> for (i, m0, m1) in specgen(nmap):
            print((i, m0, m1))
        (0, 0, 0)
        (1, 1, 1)
        (2, 2, 2)
        (3, 0, 1)
        (4, 1, 2)
        (5, 0, 2)
    
    """

    # Create lists of map ordering
    m0 = []
    m1 = []
    for lag in range(nmap):
        for i in range(nmap - lag):
            m0.append(i)
            m1.append(i + lag)
    # Iterate through spectra
    nspec = nmap * (nmap + 1) // 2
    for i in range(nspec):
        yield (i, m0[i], m1[i])

class MapDef():
    """
    The MapDef object describes the properties of a map (or maps).
    It is also used to describe the map inputs to auto and cross spectra.

    """

    def __init__(self, name, field, bandpass=None, Bl=None, fwhm_arcmin=None):
        """
        Create a new MapDef object.

        Parameters
        ----------
        name : string
            Name of the map.
        field : string
            Specifies which field the map represents. Valid options are 'T',
            'E', 'B', 'QU', or 'TQU'.
        bandpass : Bandpass object, optional
            Object describing the bandpass of the map.
        Bl : array, dtype=float, optional
            Array specifying beam window function for this map. It is assumed
            that this function starts at ell = 0 and extends up to
            ell_max = len(Bl) - 1. If Bl is defined, then it supersedes the
            `fwhm_arcmin` argument.
        fwhm_arcmin : float, optional
            Full-width at half maximum, in arc-minutes, to define a Gaussian
            beam for this map. If the `Bl` argument is specified, then it
            supersedes this value.

        """
        
        self.name = name
        assert field.upper() in ['T','E','B','QU','TQU']
        self.field = field.upper()
        self.bandpass = bandpass
        self.Bl = Bl
        self.fwhm_arcmin = fwhm_arcmin

    def __str__(self):
        return '{}_{}'.format(self.name, self.field)

    def __eq__(self, value):
        """
        MapDef objects are considered equivalent if the name and field match.
        We don't check whether the bandpasses match.
        
        """
        
        return (self.name == value.name) and (self.field == value.field)

    def beam(self, ell_max):
        """
        Returns the beam window function for this map.

        Parameters
        ----------
        ell_max : int
            Maximum ell value for the beam window function.

        Returns
        -------
        Bl : array, shape=(ell_max+1,), dtype=float
            Beam window function defined starting at ell=0 and extending up to
            ell=ell_max. If a beam is not defined for this map, the window
            function returned will be all ones.
        
        """

        # If Bl is defined, use that
        if self.Bl is not None:
            if len(self.Bl) > ell_max:
                Bl = self.Bl[0:ell_max+1].copy()
            else:
                Bl = np.zeros(ell_max + 1)
                Bl[0:len(self.Bl)] = self.Bl.copy()
        # otherwise, calculate Gaussian Bl from FWHM
        elif self.fwhm_arcmin is not None:
            ell = np.arange(ell_max + 1)
            sigma_rad = np.radians(self.fwhm_arcmin / 60) * np.sqrt(8 * np.log(2))
            Bl = np.exp(-0.5 * ell**2 * sigma_rad**2)
        # no beam defined
        else:
            Bl = np.ones(ell_max + 1)
        return Bl

class XSpec():
    """
    The XSpec object contains the full set of auto and cross spectra for a
    list of maps, and supports multiple realizations.

    """

    def __init__(self, maplist, bins, spec):
        """
        Create a new XSpec object.

        Parameters
        ----------
        maplist : list of MapDef objects
            The set of maps that define auto and cross-spectra.
        bins : array, shape=(2,nbin)
            Lower and upper edges for each ell bin. Ell bin lower edges should
            be stored in bins[0,:] and upper edges in bins[1,:].
        spec : array, shape=(nspec,nbin,nrlz)
            Array of auto and cross-spectra following vecp ordering along
            axis 0. Ell bins extend along axis 1. If there are multiple
            independent realizations of the spectra, these are provided along
            axis 2. The array can be two-dimensional if there is only one
            realization.

        """

        # Record map list and ell bins.
        self.maplist = maplist
        self.bins = bins
        # Check that spectra have the right shape.
        nmap = len(maplist)
        nspec = nmap * (nmap + 1) // 2
        assert spec.shape[0] == nspec
        nbin = bins.shape[1]
        assert spec.shape[1] == nbin
        # Expand spec array to three dimensions, if necessary.
        # Should we .copy() these arrays??
        if spec.ndim == 2:
            self.spec = spec.reshape(nspec, nbin, 1)
        else:
            self.spec = spec

    def nmap(self):
        """Returns the number of maps"""

        return len(self.maplist)

    def nspec(self):
        """Returns the number of spectra"""

        return self.spec.shape[0]

    def nbin(self):
        """Returns the number of ell bins"""

        return self.spec.shape[1]

    def nrlz(self):
        """Returns the number of sim realizations"""

        return self.spec.shape[2]

    def __add__(self, xspec):
        """
        Concatenates two XSpec objects along the realizations axis (axis 2).

        The two XSpec objects must have matching maplist and ell bins.

        """

        assert self.maplist == xspec.maplist
        assert (self.bins == xspec.bins).all()
        return XSpec(self.maplist, self.bins,
                     np.concatenate((self.spec, xspec.spec), axis=2))

    def __getitem__(self, key):
        return self.spec.__getitem__(key)

    def __setitem__(self, key, value):
        return self.spec.__setitem__(key, value)
    
    def str(self, ispec=None):
        """
        List of spectra written in string format.

        Parameters
        ----------
        ispec : int, optional
            If specified, then returns only the string describing the spectrum
            with the specified index. By default, returns a list containing
            strings for all spectra.

        Returns
        -------
        specstr : list
            A list of strings describing the spectra. An example string would
            be "map1_B x map2_E".

        """

        specstr = []
        if ispec is not None:
            (m0, m1) = mapind(self.nspec(), ispec)
            return '{} x {}'.format(self.maplist[m0], self.maplist[m1])
        else:
            for (i,m0,m1) in specgen(self.nmap()):
                specstr.append('{} x {}'.format(self.maplist[m0], self.maplist[m1]))
        return specstr

    def select(self, maplist=None, ellind=None):
        """
        Make a new XSpec object with selected maps and/or ell bins.

        Parameters
        ----------
        maplist : list of MapDef objects, optional
            List of maps to use for the new XSpec object. Defaults to None,
            which means that the new object will have the same map list as
            the existing object.
        ellind : list, optional
            List of ell bins to keep for the new XSpec object. Ell bins are
            specified by their integer index. Defaults to None, which means to
            *keep all ell bins*.

        Returns
        -------
        xspec_new : XSpec
            New XSpec object with updated maps and ell bins.

        """

        # Process maplist argument.
        if maplist is None:
            # Not a deep copy, but I don't expect anyone to change the
            # MapDef objects out from under me.
            maplist = self.maplist.copy()
            ispec = range(self.nspec())
        else:
            # Work out spec indices for new maplist.
            ispec = [specind(self.nmap(), self.maplist.index(maplist[m0]),
                             self.maplist.index(maplist[m1]))
                     for (i,m0,m1) in specgen(len(maplist))]
        # Process ellind argument.
        if ellind is None:
            ellind = range(self.nbin())

        # Return new XSpec object with selected maps, ell bins
        # Using .copy() for spectra data.
        return XSpec(maplist, self.bins[:,ellind], self.spec[ispec,:,:].copy())

class CalcSpec():
    """
    Base class for auto and cross spectrum estimators.

    This object shouldn't be used, because it doesn't actually calculate power
    spectra. Derived classes should include the following instance variables
    and methods:
    * maplist_in : instance variable containing a list of input maps, which
                   can have field = 'T', 'QU', or 'TQU'
    * bins       : instance variable listing the lower and upper edges of each
                   ell bin, with shape=(2,nbin)
    * make_maplist_out : method that provides a list of MapDef objects that
                   define the ordering of calculated spectra. This method is
                   implemented in the base class and can probably be reused by
                   derived classes.
    * nmap       : method that returns the length of the *output* maplist
    * nspec      : method that returns the number of output spectra
    * nbin       : method that returns the number of ell bins
    * calc       : method that takes a list of input maps and returns an XSpec
                   object containing the calculated output spectra

    """

    def __init__(self, maplist_in, bins, nside):
        """
        Create a new CalcSpec object.

        Parameters
        ----------
        maplist_in : list of MapDef
            This is a list that defines the maps that we will calculate auto
            and cross spectra form. These input maps should have field set to
            'T', 'QU', or 'TQU'.
        bins : array, shape=(2,nbin)
            Array containing the lower edges, in bins[0,:], and upper edges, in
            bins[1,:], of each ell bin. Following the usual python convention,
            ell bins are defined to be *inclusive* of the lower edge but
            *exclusive* of the upper edge.
        nside : int, power of 2
            Healpix NSIDE used for *all* maps. 

        """
        
        self.maplist_in = maplist_in
        self.make_maplist_out()
        self.bins = bins
        self.nside = nside

    def make_maplist_out(self):
        """
        Computes list of maps that define the output spectra.

        This function is usually called in the constructor, immediately after
        the input maplist is recorded.

        When calculating the output maplist, this function assumes that we will
        calculate all possible spectra from the input maps. So if the input
        maplist contains one 'TQU' entry, there are six output spectra: TT, EE,
        BB, TE, EB, and TB, and there are three output maps: T, E, B.

        """
        
        self.maplist_out = []
        for m in self.maplist_in:
            if m.field == 'T':
                self.maplist_out.append(MapDef(m.name, 'T', m.bandpass, m.Bl, m.fwhm_arcmin))
            elif m.field == 'QU':
                self.maplist_out.append(MapDef(m.name, 'E', m.bandpass, m.Bl, m.fwhm_arcmin))
                self.maplist_out.append(MapDef(m.name, 'B', m.bandpass, m.Bl, m.fwhm_arcmin))
            elif m.field == 'TQU':
                self.maplist_out.append(MapDef(m.name, 'T', m.bandpass, m.Bl, m.fwhm_arcmin))
                self.maplist_out.append(MapDef(m.name, 'E', m.bandpass, m.Bl, m.fwhm_arcmin))
                self.maplist_out.append(MapDef(m.name, 'B', m.bandpass, m.Bl, m.fwhm_arcmin))
            else:
                raise ValueError('input maps to CalcSpec must be T, QU, or TQU')

    def nmap(self):
        """Returns the number of maps in the output maplist"""
        
        return len(self.maplist_out)

    def nspec(self):
        """Returns the number of output spectra"""
        
        return self.nmap() * (self.nmap() + 1) // 2

    def nbin(self):
        """Returns the number of ell bins"""
        
        return self.bins.shape[1]
                
    def calc(self, maps):
        """
        Placeholder for function to calculate power spectra

        Parameters
        ----------
        maps : list
            This list should contain Healpix maps that match maplist_in. The
            Healpix maps are arrays with shape=(nmap,npix). Each maplist_in
            entry has field = 'T' (nmap=1), 'QU' (nmap=2), or 'TQU' (nmap=3).
            The npix value should match the Healpix NSIDE defined in the
            constructor.

        Returns
        -------
        spec : XSpec object
            Object containing an array of power spectra with shape
            (nspec, nbin, 1).

        """
        
        print('WARNING: CalcSpec base class shouldn''t be used!')
        assert len(maps) == len(self.maplist_in)
        return XSpec(self.maplist_out, self.bins,
                     np.zeros(shape=(self.nspec(), self.nbin(), 1)))
    

