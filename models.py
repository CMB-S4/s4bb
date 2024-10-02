"""
============================
Cross-spectrum theory models
============================

"""

import numpy as np
from util import specgen

class Model():
    """
    Base class for theory models

    """

    # Class variable defines the number of parameters in the model. This should
    # be overwritten to an appropriate value for all derived classes!
    nparameters = 0
    
    def __init__(self, maplist, wf):
        """
        Constructor for Model base class (not an actual model!)

        Parameters
        ----------
        maplist : list of MapDef objects
        wf : BPWF object

        """
        
        self.maplist = maplist
        # Check that bpwf has compatible maplist
        assert wf.maplist == maplist
        self.wf = wf

    def nmap(self):
        """Get the number of maps for this model"""
        
        return len(self.maplist)

    def nspec(self):
        """Get the number of spectra for this model"""
        
        N = self.nmap()
        return (N * (N + 1) // 2)

    def nbin(self):
        """Get the number of ell bins for this model"""
        
        return self.wf.nbin

    def nparam(self):
        """Get the number of parameters for this model"""
        
        return self.__class__.nparameters

    def param_list_to_dict(self, param_list):
        """
        Convert list of parameters to a structured dictionary

        Parameters
        ----------
        param_list : list
            List of 0 parameters in following order:

        Returns
        -------
        param_dict : dict
            Dictionary of parameters

        """
        
        # No parameters, return empty dict
        return {}

    def param_dict_to_list(self, param_dict):
        """
        Convert parameters dictionary to an ordered list

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters with the following 0 keys:

        Returns
        -------
        param_list : list
            List of 0 parameters in following order:

        """
        
        # No parameters, return empty list
        return []

    def theory_spec(self, param, m0, m1, lmax=None):
        """
        Return six theory spectra (TT,EE,BB,TE,EB,TB) for specified maps.

        Note that the m0 and m1 indexes refer to MapDef objects that specify
        both the map name and field. So if, for example, these both point to
        B-mode maps, then you might expect this function to return a BB spectrum
        only. Instead we return all six spectra because there might be leakage
        from TT->BB, EE->BB, etc, that is calculated during application of the
        bandpower window functions.

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.
        m0 : int
            Index of the first map in the cross spectrum.
        m1 : int
            Index of the second map in the cross spectrum.
        lmax : int
            Maximum ell to evaluate specta. Default is the max ell value of the
            model bandpower window functions.

        Returns
        -------
        spec : array, shape=(6,lmax+1)
            Model theory spectra (TT,EE,BB,TE,EB,TB) for the specified
            parameters.

        """

        # Check that m0,m1 are valid.
        assert m0 < self.nmap()
        assert m1 < self.nmap()
        # If parameters are supplied as a dict, convert to list.
        if type(param) == dict:
            param = self.param_dict_to_list(param)
        assert len(param) == self.nparam()
        # Get lmax
        if lmax is None:
            lmax = self.wf.lmax()
        # Empty model
        return np.zeros(shape=(6,lmax+1))
        
    def expv(self, param):
        """
        Return model expectation values for specified parameters

        Parameters
        ----------
        param : list or dict
            Model parameters in the form of an ordered list or a dict. See
            param_dict_to_list docstring for details of the ordering.

        Returns
        -------
        expval : array, shape=(nspec,nbin)
            Array containing model expectation values for all spectra in all
            ell bins.

        """

        # Allocate array for expectation values.
        expval = np.zeros(shape=(self.nspec(),self.nbin()))
        # Loop over spectra
        for (i,m0,m1) in specgen(self.nmap()):
            spec = self.theory_spec(param, m0, m1)
            # Use window functions to calculate how the six spectra couple to
            # these bandpowers.
            for (j,spectype) in enumerate(['TT','EE','BB','TE','EB','TB']):
                expval[i,:] += self.wf.expv(spec[j,:], spectype, i)
        return expval

    def select(self, maplist=None, ellind=None):
        """
        Make a new Model object with selected maps and/or ell bins.

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
        mod_new : Model
            New Model object with updated maps and ell bins.

        """

        wf_new = self.wf.select(maplist, ellind)
        return self.__class__(maplist, wf_new)

