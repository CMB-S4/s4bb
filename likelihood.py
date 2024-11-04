"""
===========
Likelihoods
===========

"""

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

    def set_bias(self, bias):
        assert bias.maplist == self.maplist
        if self.bpcm is not None:
            assert bias.nbin() == self.bpcm.nbin
        for model in self.models():
            assert bias.nbin() == model.nbin()
        self.bias = bias

    def set_bpcm(self, bpcm):
        assert bpcm.maplist == self.maplist
        if self.bias is not None:
            assert bpcm.nbin == self.bias.nbin()
        for model in self.models():
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
        for model in self.models():
            n = model.nparam()
            temp_dict = model.param_list_to_dict(param_list[i:i+n])
            for (key,val) in temp_dict.items():
                param_dict[key] = val
            i += n
        return param_dict

    def param_dict_to_list(self, param_dict):
        param_list = []
        for model in self.models():
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
            expval += self.bias
        return expval
            
    
    
