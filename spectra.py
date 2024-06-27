"""
=======================
Cross-spectral analysis
=======================

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
    The MapDef object describes the properties of a map that is included in the
    cross-spectral analysis.

    """

    def __init__(self, name, field, bandpass=None):
        """
        Create a new MapDef object.

        Parameters
        ----------
        name : string
            Name of the map.
        field : string
            Specifies which field the map represents. Valid options are 'T',
            'E', or 'B'.
        bandpass : Bandpass object, optional
            Object describing the bandpass of the map.

        """
        
        self.name = name
        assert field.upper() in ['T','E','B']
        self.field = field.upper()
        self.bandpass = bandpass

    def __str__(self):
        return '{} (type: {})'.format(self.name, self.field)

    def __eq__(self, value):
        """
        MapDef objects are considered equivalent if the name and field match.
        We don't check whether the bandpasses match.
        
        """
        
        return (self.name == value.name) and (self.field == value.field)

class XSpec():
    """
    The XSpec object is used for likelihood analysis using the set of auto and
    cross-spectra.
    
    """
    
    def __init__(self, maplist, bpcov=None, bpwf=None):
        self.maplist = maplist
        self.bpcov = bpcov
        self.bpwf = bpwf

    def __str__(self):
        desc = '[XSpec object] {} maps:'.format(len(self.maplist))
        for i in range(len(self.maplist)):
            desc += ' {}'.format(self.maplist[i].name)
        return desc
