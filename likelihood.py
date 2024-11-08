"""
===========
Likelihoods
===========

"""

import warnings
import numpy as np
from s4bb.util import vecp_to_matrix, matrix_to_vecp

class Likelihood():
    def __init__(self, maplist, bias=None, bpcm=None, models=[]):
        """
        Construct a new Likelihood object

        Parameters
        ----------
        maplist : list of MapDef objects
        bias : XSpec
        bpcm : BpCov
        models : list of Model objects

        """
        
        self.maplist = maplist
        self.bias = None
        self.bpcm = None
        self.set_models(models)        
        if bias is not None:
            self.set_bias(bias)
        if bpcm is not None:
            self.set_bpcm(bpcm)
        # Use compute_fiducial_bpcm method to set this.
        self.fiducial = None

    def set_bias(self, bias):
        assert bias.maplist == self.maplist
        if self.bpcm is not None:
            assert bias.nbin() == self.bpcm.nbin
        for model in self.models:
            assert bias.nbin() == model.nbin()
        self.bias = bias

    def set_bpcm(self, bpcm):
        assert bpcm.maplist == self.maplist
        if self.bias is not None:
            assert bpcm.nbin == self.bias.nbin()
        for model in self.models:
            assert bpcm.nbin == model.nbin()
        self.bpcm = bpcm

    def set_models(self, models):
        for model in models:
            assert model.maplist == self.maplist
            if self.bias is not None:
                assert model.nbin() == self.bias.nbin()
            if self.bpcm is not None:
                assert model.nbin() == self.bpcm.nbin
        self.models = models

    def nmap(self):
        return len(self.maplist)

    def nspec(self):
        n = self.nmap()
        return (n * (n + 1) // 2)

    def nbin(self):
        # There are a variety of ways to get nbin, some of which might not be
        # defined. They should all be consistent.
        try:
            return self.bias.nbin()
        except AttributeError:
            pass
        try:
            return self.bpcm.nbin
        except AttributeError:
            pass
        if len(self.models) > 0:
            return self.models[0].nbin()
        return None
        
    def nparam(self):
        n = 0
        for model in models:
            n += model.nparam()
        return n

    def param_names(self):
        names = []
        for model in self.models:
            for param in model.param_names():
                names.append(param)
        return names

    def param_list_to_dict(self, param_list):
        param_dict = {}
        i = 0
        for model in self.models:
            n = model.nparam()
            temp_dict = model.param_list_to_dict(param_list[i:i+n])
            for (key,val) in temp_dict.items():
                param_dict[key] = val
            i += n
        return param_dict

    def param_dict_to_list(self, param_dict):
        param_list = []
        for model in self.models:
            temp_list = model.param_dict_to_list(param_dict)
            for param in temp_list:
                param_list.append(param)
        return param_list

    def expv(self, param, include_bias=True):
        expval = np.zeros(shape=(self.nspec(), self.nbin()))
        if type(param) == dict:
            for model in self.models:
                expval += model.expv(param)
        elif type(param) == list:
            i = 0
            for model in self.models:
                n = model.nparam()
                expval += model.expv(param[i:i+n])
                i += n
        else:
            raise AttributeError('param must be list or dict')
        if include_bias:
            expval += self.bias[:,:,0]
        return expval
            
    def compute_fiducial_bpcm(self, expv, noffdiag=None, mask_noise=True):
        """
        Calculates and stores bandpower covariance matrix for chosen signal 
        model.

        Parameters
        ----------
        expv : array
            Array of bandpower expectation values with shape (N,M), where N is
            the number of spectra and M is the number of ell bins.

        Returns
        -------
        None

        """
    
        self.fiducial = {}
        # Record fiducial model bandpowers in matrix form (include noise bias)
        self.fiducial['Cf'] = vecp_to_matrix(expv + self.bias[:,:,0])
        # Also, take the square root (cholesky decomposition) in each ell bin
        self.fiducial['Cf12'] = np.permute_dims(np.linalg.cholesky(
            np.permute_dims(self.fiducial['Cf'], (2,0,1))), (1,2,0))
        # Get bandpower covariance matrix.
        try:
            # BpCov_signoi
            self.fiducial['M'] = self.bpcm.get(sig_model=expv, noffdiag=noffdiag,
                                               mask_noise=mask_noise)
        except TypeError:
            # BpCov base class
            raise RuntimeWarning('BpCov does not support signal scaling\n' +
                                 'Should use BpCov_signoi class instead')
            self.fiducial['M'] = self.bpcm.get(noffdiag=noffdiag)
        # Calculate inverse bandpower covariance matrix.
        self.fiducial['Minv'] = np.linalg.inv(self.fiducial['M'])

    def hl_likelihood(self, expv, data):
        """
        Calculates Hamimeche-Lewis likelihood for specified model expectation
        values and data.

        Parameters
        ----------
        expv : array
            Array of bandpower expectation values with shape (N,M), where N is
            the number of spectra and M is the number of ell bins.
        data : array
            Array of data bandpowers with same shape as expv.

        Returns
        -------
        logL : float
            -2 * log(likelihood). See equations 47-49 of Hamimeche-Lewis (2008),
            PRD 77, 103013.

        """

        # Convert expv and data from vecp to matrix form.
        C = vecp_to_matrix(expv)
        Chat = vecp_to_matrix(data)
        nmap = C.shape[0]
        nspec = nmap * (nmap + 1) // 2
        # Transform expv and data to Gaussian-like quantity.
        nbin = self.nbin()
        X = np.zeros(C.shape)
        for i in range(nbin):
            Cn12 = np.linalg.cholesky(np.linalg.inv(C[:,:,i]))
            (eigval, eigvec) = np.linalg.eigh(np.transpose(Cn12) @ Chat[:,:,i] @ Cn12)
            g = np.sign(eigval - 1.0) * np.sqrt(2 * (eigval - np.log(eigval) - 1.0))
            X[:,:,i] = (self.fiducial['Cf12'][:,:,i] @
                        (eigvec @ np.diag(g) @ np.transpose(eigvec)) @
                        np.transpose(self.fiducial['Cf12'][:,:,i]))
        # Convert X to vecp ordering and then concatenate ell bins to get a
        # long vector that matches our bpcm.
        Xv = np.reshape(matrix_to_vecp(X), (nspec*nbin,), order='F')
        # Calculate -2*log(L) as chi^2
        logL = Xv @ self.fiducial['Minv'] @ Xv
        return logL
