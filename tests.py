"""
==========
Unit Tests
==========

"""

import unittest
import numpy as np

from spectra import MapDef, specind, mapind, specgen
from bandpass import Bandpass
from bpwf import BPWF
from bpcov import BpCov

class SpectraTest(unittest.TestCase):
    """
    Unit tests for spectra.py

    """
    
    def setUp(self):
        # Five maps with a mix of T, E, B
        self.maps = [MapDef('m0_T', 'T'),
                     MapDef('m1_E', 'E'),
                     MapDef('m2_B', 'B'),
                     MapDef('m3_E', 'E'),
                     MapDef('m4_B', 'B')]
        self.nmap = len(self.maps)
        # Five maps yields 15 spectra... specify in vecp ordering
        self.spec = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                     (0, 1), (1, 2), (2, 3), (3, 4),
                     (0, 2), (1, 3), (2, 4),
                     (0, 3), (1, 4),
                     (0, 4)]
        self.nspec = len(self.spec)

    def test_specind(self):
        """Test specind function"""

        for i in range(self.nspec):
            self.assertEqual(i, specind(self.nmap, self.spec[i][0],
                                             self.spec[i][1]))

    def test_mapind(self):
        """Test mapind function"""

        for i in range(self.nspec):
            (m0, m1) = mapind(self.nspec, i)
            self.assertEqual(self.spec[i], (m0, m1))

    def test_specgen(self):
        """Test specgen generator"""

        for (i, m0, m1) in specgen(self.nmap):
            self.assertEqual(self.spec[i], (m0, m1))

    def test_MapDef(self):
        """Test MapDef class"""

        # Equality requires same name *and* field
        self.assertEqual(self.maps[0], MapDef('m0_T', 'T'))
        self.assertNotEqual(self.maps[0], MapDef('m0_T', 'E'))
        self.assertNotEqual(self.maps[0], MapDef('m1_T', 'T'))

class BandpassTest(unittest.TestCase):
    """
    Unit tests for bandpass.py

    """

    def test_deltafn(self):
        """Test delta-fn bandpass"""

        nu0 = 100.0 # GHz
        bp = Bandpass.deltafn(nu0)
        self.assertEqual(1.0, bp.bandpass_integral(lambda x: 1.0))
        self.assertEqual(nu0, bp.nu_eff())
        # Check the CMB unit conversion
        from bandpass import GHz, Tcmb, h, k, c
        x = h * nu0 * GHz / (k * Tcmb)
        conv = (2 * k**3 * Tcmb**2 / (h**2 * c**2) * x**4 * np.exp(x) /
                (np.exp(x) - 1)**2)
        self.assertEqual(conv, bp.cmb_unit_conversion())

    def test_tophat(self):
        """Test tophat bandpass"""

        nu0 = 100.0 # GHz
        nu1 = 120.0
        bp = Bandpass.tophat(nu0, nu1, RJ=False)
        self.assertEqual(1.0, bp.bandpass_integral(lambda x: 1.0))
        self.assertEqual((nu0 + nu1) / 2, bp.nu_eff())
        for nu in np.linspace(nu0, nu1, 10):
            self.assertEqual(1.0 / (nu1 - nu0), bp.fn(nu))

class BpwfTest(unittest.TestCase):
    """
    Unit tests for bpwf.py

    """

    def setUp(self):
        # BPWF object with three maps (T, E, B) and tophat window functions.
        self.maplist = [MapDef('m0_T', 'T'),
                        MapDef('m1_E', 'E'),
                        MapDef('m2_B', 'B')]
        self.wf = BPWF.tophat(self.maplist, [10, 20, 30, 40, 50, 60],
                              lmin=0, lmax=100)
        self.nbin = self.wf.nbin

    def test_expv(self):
        """Test BPWF.expv method"""

        # map0 x map0 should be TT only
        self.assertTrue(all(self.wf.expv('TT', 0, 0, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['EE','BB','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(spectype, 0, 0, lambda x: 1) == np.zeros(self.nbin)))
        # map1 x map1 should be EE only
        self.assertTrue(all(self.wf.expv('EE', 1, 1, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','BB','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(spectype, 1, 1, lambda x: 1) == np.zeros(self.nbin)))
        # map2 x map2 should be BB only
        self.assertTrue(all(self.wf.expv('BB', 2, 2, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','EE','TE','EB','TB']:
            self.assertTrue(all(self.wf.expv(spectype, 2, 2, lambda x: 1) == np.zeros(self.nbin)))
        # map0 x map2 should be TB only
        self.assertTrue(all(self.wf.expv('TB', 0, 2, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','EE','BB','TE','EB']:
            self.assertTrue(all(self.wf.expv(spectype, 0, 2, lambda x: 1) == np.zeros(self.nbin)))

    def test_select(self):
        """Test BPWF select method"""

        # First, try downselecting maps -- keep E and B only
        wfnew = self.wf.select([self.maplist[1], self.maplist[2]], None)
        # map0 x map0 should be EE only
        self.assertTrue(all(wfnew.expv('EE', 0, 0, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','BB','TE','EB','TB']:
            self.assertTrue(all(wfnew.expv(spectype, 0, 0, lambda x: 1) == np.zeros(self.nbin)))
        # map1 x map1 should be BB only
        self.assertTrue(all(wfnew.expv('BB', 1, 1, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','EE','TE','EB','TB']:
            self.assertTrue(all(wfnew.expv(spectype, 1, 1, lambda x: 1) == np.zeros(self.nbin)))
        # map0 x map1 should be EB only
        self.assertTrue(all(wfnew.expv('EB', 0, 1, lambda x: 1) == np.ones(self.nbin)))
        for spectype in ['TT','EE','BB','TE','TB']:
            self.assertTrue(all(wfnew.expv(spectype, 0, 1, lambda x: 1) == np.zeros(self.nbin)))

        # Next, try downselecting ell bins -- keep bins 2 and 3 only
        keep = [2, 3]
        wfnew = self.wf.select(None, keep)
        self.assertEqual(len(keep), wfnew.nbin)
        self.assertTrue(all(self.wf.ell_eff('TT', 0, 0)[keep] == wfnew.ell_eff('TT', 0, 0)))

if __name__ == '__main__':
    unittest.main()



