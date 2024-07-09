"""
==========
Unit Tests
==========

"""

import unittest
import numpy as np

from spectra import MapDef, specind, mapind, specgen
from bandpass import Bandpass

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
    
if __name__ == '__main__':
    unittest.main()



