"""
=================
Utility functions
=================

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

    def copy(self, update_field=None):
        """
        Returns a copy of the MapDef object.

        Parameters
        ----------
        update_field : str, optional
            By default, the returned MapDef object will have the same field
            type as the current object. Use this optional argument to update to
            a new field type.

        Returns
        -------
        map_new : MapDef object
            Copy of the current object (not a deep copy!).

        """

        if update_field is not None:
            return MapDef(self.name, update_field, bandpass=self.bandpass,
                          Bl=self.Bl, fwhm_arcmin=self.fwhm_arcmin)
        else:
            return MapDef(self.name, self.field, bandpass=self.bandpass,
                          Bl=self.Bl, fwhm_arcmin=self.fwhm_arcmin)
    
    def beam(self, ell_max, Bl_min=0.0):
        """
        Returns the beam window function (Bl) for this map.

        Parameters
        ----------
        ell_max : int
            Maximum ell value for the beam window function.
        Bl_min : float, optional
            Minimum value for the beam window function. In the case of large
            beam sizes, Bl can reach extremely small values at high ell, which
            leads to problems when dividing by Bl to correct bandpowers. This
            argument allows you to specify a floor to the beam window function.
            Default value is 0, i.e. no floor.

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
            sigma_rad = np.radians(self.fwhm_arcmin / 60) / np.sqrt(8 * np.log(2))
            Bl = np.exp(-0.5 * ell**2 * sigma_rad**2)
        # no beam defined
        else:
            Bl = np.ones(ell_max + 1)
        # Apply floor to Bl
        Bl[Bl < Bl_min] = Bl_min
        # Done
        return Bl
