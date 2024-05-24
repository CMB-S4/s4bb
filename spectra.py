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

class MapDef():
    """
    The MapDef object describes the properties of a map that is included in the
    cross-spectral analysis.

    """

    def __init__(self, name, field, bandpass):
        self.name = name
        assert field.upper() in ['T','E','B','LB']
        self.field = field.upper()
        self.bandpass = bandpass

class XSpec():
    """
    The XSpec object is used for likelihood analysis using the set of auto and
    cross-spectra.
    
    """
    
    def __init__(self, maplist, ellbins, bpcov=None, bpwf=None):
        self.maplist = maplist
        self.ellbins = ellbins
        self.bpcov = bpcov
        self.bpwf = bpwf
        
