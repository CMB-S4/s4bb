"""
=============
Power spectra
=============

"""

import numpy as np
import healpy as hp
# Check whether NaMaster is installed
try: import pymaster as nmt
except ImportError: nmt = None

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

def fix_map(map_):
    """
    Convert NaN, Inf, and hp.UNSEEN pixels to zero.

    Operates on the input map(s) in place.

    Parameters
    ----------
    map_ : array
        Array containing one or more maps.

    """

    map_[np.isnan(map_)] = 0.0
    map_[np.isinf(map_)] = 0.0
    map_[map_ == hp.UNSEEN] = 0.0
    
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

    def __init__(self, maplist_in, apod, nside, bins, use_Dl=True, Bl_min=0.0):
        """
        Create a new CalcSpec object.

        Parameters
        ----------
        maplist_in : list of MapDef
            This is a list that defines the maps that we will calculate auto
            and cross spectra form. These input maps should have field set to
            'T', 'QU', or 'TQU'.
        apod : Healpix map or list of Healpix maps
            Apodization that will be used to weight maps before transform.
            If a single Healpix map is supplied, then the same apodization
            will be used for all maps. The alternative is to supply a list of
            apodization maps, one for each entry in maplist_in.
        nside : int, power of 2
            Healpix NSIDE used for *all* maps.
        bins : array, shape=(2,nbin)
            Array containing the lower edges, in bins[0,:], and upper edges, in
            bins[1,:], of each ell bin. Following the usual python convention,
            ell bins are defined to be *inclusive* of the lower edge but
            *exclusive* of the upper edge.
        use_Dl : bool, optional
            By default, calculates Dl = l*(l+1)*Cl/(2*pi). Set argument to
            false to calculate Cl instead.
        Bl_min : float, optional
            If specified, sets the minimum value for Bl to avoid divide-by-zero
            errors. Default value is 0.

        """
        
        self.maplist_in = maplist_in
        self.make_maplist_out()
        # If apod is a numpy array, then this same apodization should be
        # applied to all maps.
        try:
            apod.shape # throws an exception if apod is already a list
            self.apod = [apod] * self.nmap() # convert to list by repetition
        except:
            self.apod = apod
        self.bins = bins
        self.nside = nside
        self.use_Dl = use_Dl
        self.Bl_min = Bl_min

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
                self.maplist_out.append(m.copy())
            elif m.field == 'QU':
                self.maplist_out.append(m.copy(update_field='E'))
                self.maplist_out.append(m.copy(update_field='B'))
            elif m.field == 'TQU':
                self.maplist_out.append(m.copy(update_field='T'))
                self.maplist_out.append(m.copy(update_field='E'))
                self.maplist_out.append(m.copy(update_field='B'))
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
        return XSpec(self.maplist_out, self.bins,
                     np.zeros(shape=(self.nspec(), self.nbin(), 1)))

class CalcSpec_healpy(CalcSpec):
    """
    Calculate power spectra using healpy tools.

    """

    def calc(self, maps):
        """
        Calculates auto and cross spectra for apodized maps.

        Parameters
        ----------
        maps : list of Healpix maps

        Returns
        -------
        spec : XSpec object

        """

        # Process maps:
        #   - Convert NaN, inf, hp.UNSEEN values to 0
        #   - Make sure that they match mapdef_in
        #   - Apply apodization and calculate alms
        alm = []
        for i in range(len(maps)):
            # Fixing NaN, inf, hp.UNSEEN values
            # Is this a waste of time? Should we rely on the user to do it?
            fix_map(maps[i])
            # Which alms to calculate depends on the input map field(s).
            if self.maplist_in[i].field == 'T':
                t_lm = hp.map2alm(maps[i] * self.apod[i])
                # Rough fsky correction
                t_lm = t_lm / np.sqrt(np.mean(self.apod[i]**2))
                alm.append(t_lm)
            elif self.maplist_in[i].field == 'QU':
                assert (maps[i].shape[0] == 2)
                # Combine QU maps with an empty T map
                tqu = np.zeros(shape=(3,maps[i].shape[1]))
                tqu[1:,:] = maps[i] * self.apod[i]
                teb_lm = hp.map2alm(tqu, pol=True)
                # Rough fsky correction
                teb_lm = teb_lm / np.sqrt(np.mean(self.apod[i]**2))
                alm.append(teb_lm[1])  # E_lm
                alm.append(teb_lm[2])  # B_lm
            elif self.maplist_in[i].field == 'TQU':
                assert (maps[i].shape[0] == 3)
                teb_lm = hp.map2alm(maps[i] * self.apod[i], pol=True)
                # Rough fsky correction
                teb_lm = teb_lm / np.sqrt(np.mean(self.apod[i]**2))
                alm.append(teb_lm[0]) # T_lm
                alm.append(teb_lm[1]) # E_lm
                alm.append(teb_lm[2]) # B_lm
            else:
                raise ValueError('input maps to CalcSpec must be T, QU, or TQU')
        # Calculate power spectra:
        #   - Loop over output spectra and calculate Cl from alms
        #   - Convert to Dl, if desired
        #   - Divide by Bl
        #   - Apply ell binning
        spec = np.zeros(shape=(self.nspec(),self.nbin(),1))
        ell = np.arange(3 * self.nside)
        Dlconv = ell * (ell + 1) / (2 * np.pi)
        for (i,m0,m1) in specgen(self.nmap()):
            Cl = hp.alm2cl(alm[m0], alm[m1])
            # Cl -> Dl conversion but keep same variable name
            if self.use_Dl: Cl = Cl * Dlconv
            # Divide out beam window functions
            Cl = Cl / self.maplist_out[m0].beam(len(Cl) - 1, Bl_min=self.Bl_min)
            Cl = Cl / self.maplist_out[m1].beam(len(Cl) - 1, Bl_min=self.Bl_min)
            # Apply ell binning
            for j in range(self.nbin()):
                spec[i,j,0] = Cl[self.bins[0,j]:self.bins[1,j]].mean()
        # Return spectra as XSpec object.
        return XSpec(self.maplist_out, self.bins, spec)

class CalcSpec_namaster(CalcSpec):
    """
    Calculate power spectra using NaMaster
        
    """

    def __init__(self, maplist_in, apod, nside, bins, use_Dl=True,
                 Bl_min=0.0, pure_B=False):
        """
        Create a new CalcSpec_namaster object.
        
        """

        # Raise an error if NaMaster is not installed.
        if nmt is None:
            raise ImportError('NaMaster is not installed.')

        # Base class constructor handles some initial set up.
        super().__init__(maplist_in, apod, nside, bins,
                         use_Dl=use_Dl, Bl_min=Bl_min)

        # If pure_B is a single boolean value, then repeat it for each map.
        try:
            len(pure_B) # will thrown an exception if pure_B is a boolean
            self.pure_B = pure_B
        except:
            self.pure_B = [pure_B] * len(maplist_in)

        # NaMaster binning operator
        self.nmt_bin = nmt.NmtBin.from_edges(self.bins[0,:], self.bins[1,:],
                                             is_Dell=self.use_Dl)
            
        # Set up NaMaster workspaces for power spectrum calculation.
        # First, we need to generate spin-0 and spin-2 NmtField objects.
        # Also gets the indexes to map from NmtField to maplist_out.
        (fields, map_index) = self.make_fields()
        # Then we need to make a list of NmtWorkspace for each auto or cross
        # spectrum calculation, but also record how the output of each
        # NmtWorkspace maps to our output spectrum ordering.
        self.workspaces = []
        self.spec_index = []
        for (i,m0,m1) in specgen(len(fields)):
            # Create workspace and compute coupling matrix.
            ws = nmt.NmtWorkspace.from_fields(fields[m0], fields[m1], self.nmt_bin)
            # Figure out the mapping from NaMaster ordering to our spectral
            # ordering.
            idx0 = map_index[m0]
            idx1 = map_index[m1]
            if (len(idx0) == 1) and (len(idx1) == 1):
                # spin-0 x spin-0
                ispec = [specind(self.nmap(), idx0[0], idx1[0])]
            elif (len(idx0) == 1) and (len(idx1) == 2):
                # spin-0 x spin-2
                ispec = [specind(self.nmap(), idx0[0], j) for j in idx1]
            elif (len(idx0) == 2) and (len(idx1) == 1):
                # spin-2 x spin-0
                ispec = [specind(self.nmap(), j, idx1[0]) for j in idx0]
            elif (len(idx0) == 2) and (len(idx1) == 2):
                # spin-2 x spin-2
                ispec = [specind(self.nmap(), idx0[0], idx1[0]),
                         specind(self.nmap(), idx0[0], idx1[1]),
                         specind(self.nmap(), idx0[1], idx1[0]),
                         specind(self.nmap(), idx0[1], idx1[1])]
            else:
                raise ValueError('NmtField with wrong number of entries!')
            # Store results.
            self.workspaces.append(ws)
            self.spec_index.append(ispec)

    def make_fields(self, maps=None):
        """
        Builds a list of NmtField objects corresponding to the input maps.

        """
        
        # If maps are not provided, expand maps argument to match length
        # of self.maplist_in
        if maps is None: maps = [None] * len(self.maplist_in)

        # Work through maplist_in and build a list of fields as well as the
        # indexing needed to match NmtFields with maplist_out.
        fields = []
        map_index = []
        counter = 0
        for i in range(len(self.maplist_in)):
            # Need to separately handle cases where maps[i] is None
            if self.maplist_in[i].field == 'T':
                # Add a spin-0 field to list.
                if maps[i] is not None:
                    T = nmt.NmtField(self.apod[i], [maps[i]], spin=0,
                                     beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                     lmax=self.nmt_bin.lmax,
                                     lmax_mask=self.nmt_bin.lmax)
                else:
                    T = nmt.NmtField(self.apod[i], None, spin=0,
                                     beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                     lmax=self.nmt_bin.lmax,
                                     lmax_mask=self.nmt_bin.lmax)
                fields.append(T)
                map_index.append([counter])
                counter += 1
            elif self.maplist_in[i].field == 'QU':
                # Add a spin-2 field to list.
                if maps[i] is not None:
                    QU = nmt.NmtField(self.apod[i], [maps[i][0], maps[i][1]], spin=2,
                                      beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                      lmax=self.nmt_bin.lmax,
                                      lmax_mask=self.nmt_bin.lmax,
                                      purify_b=self.pure_B[i])
                else:
                    QU = nmt.NmtField(self.apod[i], None, spin=2,
                                      beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                      lmax=self.nmt_bin.lmax,
                                      lmax_mask=self.nmt_bin.lmax,
                                      purify_b=self.pure_B[i])
                fields.append(QU)
                map_index.append([counter, counter + 1])
                counter += 2              
            elif self.maplist_in[i].field == 'TQU':
                # Need to add both spin-0 and spin-2 fields.
                if maps[i] is not None:
                    T = nmt.NmtField(self.apod[i], [maps[i][0]], spin=0,
                                     beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                     lmax=self.nmt_bin.lmax,
                                     lmax_mask=self.nmt_bin.lmax)
                    QU = nmt.NmtField(self.apod[i], [maps[i][1], maps[i][2]], spin=2, 
                                      beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                      lmax=self.nmt_bin.lmax,
                                      lmax_mask=self.nmt_bin.lmax,
                                      purify_b=self.pure_B[i])
                else:
                    T = nmt.NmtField(self.apod[i], None, spin=0,
                                     beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                     lmax=self.nmt_bin.lmax,
                                     lmax_mask=self.nmt_bin.lmax)
                    QU = nmt.NmtField(self.apod[i], None, spin=2, 
                                      beam=self.maplist_in[i].beam(self.nmt_bin.lmax, Bl_min=self.Bl_min),
                                      lmax=self.nmt_bin.lmax,
                                      lmax_mask=self.nmt_bin.lmax,
                                      purify_b=self.pure_B[i])                    
                fields.append(T)
                map_index.append([counter])
                fields.append(QU)
                map_index.append([counter + 1, counter + 2])
                counter += 3
            else:
                raise ValueError('input maps to CalcSpec must be T, QU, or TQU')
        # Return both lists
        return (fields, map_index)

    def calc(self, maps):
        """
        Calculates auto and cross spectra for apodized maps.

        Parameters
        ----------
        maps : list of Healpix maps

        Returns
        -------
        spec : XSpec object

        """

        # Convert maps into NmtField objects.
        (fields, map_index) = self.make_fields(maps)
        # Calculate spectra using workspaces to decouple spectra.
        spec = np.zeros(shape=(self.nspec(),self.nbin(),1))
        for (i,m0,m1) in specgen(len(fields)):
            cl_coupled = nmt.compute_coupled_cell(fields[m0], fields[m1])
            cl_decoupled = self.workspaces[i].decouple_cell(cl_coupled)
            # Rearrange NaMaster spectra to follow our spectra ordering.
            for j in range(len(self.spec_index[i])):
                spec[self.spec_index[i][j],:,0] = cl_decoupled[j,:]
        # Return spectra as XSpec object.
        return XSpec(self.maplist_out, self.bins, spec)
