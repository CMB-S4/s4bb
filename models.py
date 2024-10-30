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

    def param_names(self):
        """Get list of parameter names"""

        return []
    
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
        return Model(maplist, wf_new)

class Model_cmb(Model):
    """
    CMB spectra with r and Alens parameters

    """

    # Two parameter model: r, Alens
    nparameters = 2

    def __init__(self, maplist, wf, Cl_unlens, Cl_lens, Cl_tensor, lmin=0):
        """
        Constructor

        Parameters
        ----------
        maplist : list of MapDef objects
        wf : BPWF object
        Cl_unlens : array
        Cl_lens : array
        Cl_tensor : array

        """

        # Invoke base class constructor
        super().__init__(maplist, wf)
        # Each of the Cl inputs should either contain four (TT,EE,BB,TE) or
        # six (TT,EE,BB,TE,EB,TB) spectra. If lmin > 0, pad the beginning of
        # the spectra with zeros.
        # Unlensed CMB spectra
        self.Cl_unlens = np.zeros(shape=(6,lmin+Cl_unlens.shape[1]))
        if Cl_unlens.shape[0] == 4:
            self.Cl_unlens[0:4,lmin:] = Cl_unlens
        elif Cl_unlens.shape[0] == 6:
            self.Cl_unlens[:,lmin:] = Cl_unlens
        else:
            raise ValueError('Cl_unlens must contain 4 or 6 spectra')
        # Lensed CMB spectra
        self.Cl_lens = np.zeros(shape=(6,lmin+Cl_lens.shape[1]))
        if Cl_lens.shape[0] == 4:
            self.Cl_lens[0:4,lmin:] = Cl_lens
        elif Cl_lens.shape[0] == 6:
            self.Cl_lens[:,lmin:] = Cl_lens
        else:
            raise ValueError('Cl_lens must contain 4 or 6 spectra')
        # Tensor CMB spectra
        self.Cl_tensor = np.zeros(shape=(6,lmin+Cl_tensor.shape[1]))
        if Cl_tensor.shape[0] == 4:
            self.Cl_tensor[0:4,lmin:] = Cl_tensor
        elif Cl_tensor.shape[0] == 6:
            self.Cl_tensor[:,lmin:] = Cl_tensor
        else:
            raise ValueError('Cl_tensor must contain 4 or 6 spectra')

    def param_names(self):
        """Get list of parameter names"""

        return ['r', 'Alens']

    def param_list_to_dict(self, param_list):
        """
        Convert list of parameters to a structured dictionary

        Parameters
        ----------
        param_list : list
            List of 2 parameters in following order: r, Alens

        Returns
        -------
        param_dict : dict
            Dictionary of parameters

        """
        
        param_dict = {'r': param_list[0],
                      'Alens': param_list[1]}
        return param_dict

    def param_dict_to_list(self, param_dict):
        """
        Convert parameters dictionary to an ordered list

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameters with the following 2 keys: 'r', 'Alens'

        Returns
        -------
        param_list : list
            List of 2 parameters in following order: r, Alens

        """
        
        param_list = [param_dict['r'], param_dict['Alens']]
        return param_list

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
        # Allocate array for spectra
        spec = np.zeros(shape=(6,lmax+1))
        # Check whether theory spectra extend all the way to lmax
        N = min(lmax + 1, self.Cl_unlens.shape[1],
                self.Cl_lens.shape[1], self.Cl_tensor.shape[1])
        # If m0 or m1 is a lensing template, then the model will contain
        # lensing B modes only.
        if self.maplist[m0].lensing_template or self.maplist[m1].lensing_template:
            spec[2,0:N] = param[1] * self.Cl_lens[2,0:N]
        else:
            # Combine lensed and unlensed CMB spectra using Alens parameter
            spec[:,0:N] = ((1 - param[1]) * self.Cl_unlens[:,0:N] +
                           param[1] * self.Cl_lens[:,0:N])
            # Add in tensor CMB spectra using r parameter
            spec[:,0:N] += param[0] * self.Cl_tensor[:,0:N]
        # Done
        return spec

    def select(self, maplist=None, ellind=None):
        """
        Make a new Model_cmb object with selected maps and/or ell bins.

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
        mod_new : Model_cmb
            New Model_cmb object with updated maps and ell bins.

        """

        wf_new = self.wf.select(maplist, ellind)
        return Model_cmb(maplist, wf_new, self.Cl_unlens, self.Cl_lens,
                         self.Cl_tensor)
