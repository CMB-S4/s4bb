"""
==========
Unit Tests
==========

"""

import unittest
import numpy as np
import healpy as hp

from spectra import specind, mapind, specgen, MapDef, XSpec
from spectra import CalcSpec, CalcSpec_healpy, CalcSpec_namaster
from bandpass import Bandpass
from bpwf import BPWF
from bpcov import BpCov

# Some of the tests involve generating maps, calculating power spectra, and
# checking that we recover the input spectra. This involves sample variance,
# so we need to specify a tolerance for the test, expressed here in units of
# sigma.
TOL = 5.0

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
        # Lower and upper edges for three ell bins
        self.bins = np.array([[10, 20, 30], [20, 30, 40]])
        self.nbin = self.bins.shape[1]

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

    def test_XSpec(self):
        """Test XSpec class"""

        # Try concatenating two sets of spectra.
        nrlz1 = 4
        spec1 = np.ones((self.nspec, self.nbin, nrlz1))
        xspec1 = XSpec(self.maps, self.bins, spec1)
        nrlz2 = 2
        spec2 = 4 * np.ones((self.nspec, self.nbin, nrlz2))
        xspec2 = XSpec(self.maps, self.bins, spec2)
        xspec3 = xspec1 + xspec2
        self.assertEqual(xspec3.nmap(), self.nmap)
        self.assertEqual(xspec3.nspec(), self.nspec)
        self.assertEqual(xspec3.nbin(), self.nbin)
        self.assertEqual(xspec3.nrlz(), nrlz1 + nrlz2)
        self.assertTrue((xspec3.spec[:,:,0:nrlz1] == spec1).all())
        self.assertTrue((xspec3.spec[:,:,nrlz1:] == spec2).all())

        # Test setitem, getitem
        xspec3[0,1,2] = 37.0
        self.assertEqual(xspec3[0,1,2], 37)
        self.assertEqual(xspec3.spec[0,1,2], 37)

        # Test select
        # Downselect from 3 maps (6 spectra) to 2 maps (3 spectra)
        spec4 = np.ones(shape=(6,self.nbin,1))
        spec4[0,:,:] = 2.0**2 # m0 x m0
        spec4[1,:,:] = 3.0**2 # m1 x m1
        spec4[2,:,:] = 5.0**2 # m2 x m2
        spec4[3,:,:] = 2.0 * 3.0 # m0 x m1
        spec4[4,:,:] = 3.0 * 5.0 # m1 x m2
        spec4[5,:,:] = 2.0 * 5.0 # m0 x m2
        xspec4 = XSpec(self.maps[0:3], self.bins, spec4)
        xspec5 = xspec4.select([self.maps[2], self.maps[0]], None)
        self.assertEqual(xspec5.nmap(), 2)
        self.assertEqual(xspec5.nbin(), xspec4.nbin())
        self.assertEqual(xspec5.nrlz(), xspec4.nrlz())
        self.assertTrue((xspec5[0,:,:] == xspec4[2,:,:]).all())
        self.assertTrue((xspec5[1,:,:] == xspec4[0,:,:]).all())
        self.assertTrue((xspec5[2,:,:] == xspec4[5,:,:]).all())

    def test_CalcSpec(self):
        """Test CalcSpec base class"""

        m0 = MapDef('m0', 'T')
        m1 = MapDef('m1', 'QU')
        m2 = MapDef('m2', 'TQU')
        nside = 128
        apod = np.ones(hp.nside2npix(nside))
        cs = CalcSpec([m0, m1, m2], apod, nside, self.bins)
        # Check that maplist_out is as expected.
        self.assertEqual(cs.nmap(), 6)
        self.assertEqual(cs.maplist_out[0], MapDef('m0', 'T'))
        self.assertEqual(cs.maplist_out[1], MapDef('m1', 'E'))
        self.assertEqual(cs.maplist_out[2], MapDef('m1', 'B'))
        self.assertEqual(cs.maplist_out[3], MapDef('m2', 'T'))
        self.assertEqual(cs.maplist_out[4], MapDef('m2', 'E'))
        self.assertEqual(cs.maplist_out[5], MapDef('m2', 'B'))

    def test_CalcSpec_healpy(self):
        """Test CalcSpec_healpy"""

        # Check that we get the sensible power spectrum measurements.
        m0 = MapDef('m0', 'TQU')
        Cl = [8.0, 4.0, 1.0, 2.0] # TT, EE, BB, TE; white spectra
        # Try three values of NSIDE
        for nside in [64,128,256]:
            # Set up CalcSpec_healpy object
            apod = np.ones(hp.nside2npix(nside))
            bins = np.array([[10], [2 * nside]]) # one big ell bin
            cs = CalcSpec_healpy([m0], apod, nside, bins, use_Dl=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Check that we get sensible power spectrum measurements with different
        # sky cuts / apodization masks. Note that CalcSpec_healpy does not
        # include a pure-B estimator, so for this test we will set BB=0 and
        # just check the output TT, EE, and TE power.
        Cl = [8.0, 4.0, 0.0, 2.0] # TT, EE, BB, TE; white spectra
        nside = 128
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        (theta, phi) = hp.pix2ang(nside, range(hp.nside2npix(nside)))
        apod_options = []
        apod_options.append((theta < np.pi / 4).astype(float))
        apod_options.append((theta < np.pi / 6).astype(float))
        apod_options.append((theta < np.pi / 8).astype(float))
        apod_options.append((theta / np.pi)**2)
        apod_options.append((2 * (theta - np.pi / 2) / np.pi)**2)
        for apod in apod_options:
            # Set up CalcSpec_healpy object
            cs = CalcSpec_healpy([m0], apod, nside, bins, use_Dl=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            fsky = np.mean(apod**2)**2 / np.mean(apod**4) # effective fsky for mask
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside) / np.sqrt(fsky))

    def test_CalcSpec_namaster(self):
        """Test CalcSpec_namaster"""

        # Check that we get the sensible power spectrum measurements.
        m0 = MapDef('m0', 'TQU')
        Cl = [8.0, 4.0, 1.0, 2.0] # TT, EE, BB, TE; white spectra
        # Try three values of NSIDE
        for nside in [64,128,256]:
            # Set up CalcSpec_namaster object
            apod = np.ones(hp.nside2npix(nside))
            bins = np.array([[10], [2 * nside]]) # one big ell bin
            cs = CalcSpec_namaster([m0], apod, nside, bins, use_Dl=False, pure_B=False)
            # Generate input maps
            tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                             nside, new=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Test that we properly handle different combinations of T, QU, or TQU
        # input maps.
        m0 = MapDef('m0', 'TQU')
        m1 = MapDef('m1', 'T')
        m2 = MapDef('m2', 'QU')
        # Just work at one value of NSIDE
        nside = 128
        apod = np.ones(hp.nside2npix(nside))
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        # Remake TQU map at this NSIDE
        tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                         nside, new=True)
        # Independent TQU sim with same input spectrum.
        tqu2 = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                          nside, new=True)
        t = tqu2[0,:]
        qu = tqu2[1:,:]
        # T only ------
        cs = CalcSpec_namaster([m1], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([t])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        # QU only -----
        cs = CalcSpec_namaster([m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([qu])
        self.assertTrue(np.abs(Cl[1] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(spec[2,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        # T, QU -------
        cs = CalcSpec_namaster([m1, m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([t, qu])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))        
        # QU, T -------
        cs = CalcSpec_namaster([m2, m1], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([qu, t])
        self.assertTrue(np.abs(Cl[1] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[0] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(spec[3,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        # TQU, T, QU --
        cs = CalcSpec_namaster([m0, m1, m2], apod, nside, bins, use_Dl=False, pure_B=False)
        spec = cs.calc([tqu, t, qu])
        self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[0] - spec[3,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(Cl[1] - spec[4,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(Cl[2] - spec[5,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[6,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[7,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[8,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(Cl[3] - spec[9,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside))
        self.assertTrue(np.abs(spec[10,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[11,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[12,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1]) / (2 * nside))
        self.assertTrue(np.abs(spec[13,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[14,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[15,0,0]) < TOL * Cl[0] / (2 * nside))
        self.assertTrue(np.abs(spec[16,0,0]) < TOL * Cl[1] / (2 * nside))
        self.assertTrue(np.abs(spec[17,0,0]) < TOL * Cl[2] / (2 * nside))
        self.assertTrue(np.abs(spec[18,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1]) / (2 * nside))
        self.assertTrue(np.abs(spec[19,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside))
        self.assertTrue(np.abs(spec[20,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside))

        # Check that we get sensible power spectrum measurements with different
        # sky cuts / apodization masks. Turn on the NaMaster pure-B estimator
        # to deal with E->B leakage. Keep BB << EE so that we don't have to
        # worry about B->E.
        Cl = [8.0, 4.0, 0.1, 2.0] # TT, EE, BB, TE; white spectra
        m0 = MapDef('m0', 'TQU')
        nside = 128
        bins = np.array([[10], [2 * nside]]) # one big ell bin
        # Generate input maps
        tqu = hp.synfast((Cl * np.ones(shape=(3*nside,4))).transpose(),
                        nside, new=True)
        # Try some different apodizations. These should all be smooth enough
        # to work well with pure-B estimator.
        (theta, phi) = hp.pix2ang(nside, range(hp.nside2npix(nside)))
        apod_options = []
        apod_options.append(np.cos(theta)**2)
        apod_options.append(np.cos(2 * theta)**2)
        apod_options.append(np.cos(3 * theta)**2)
        apod_options.append(np.cos(4 * theta)**2)
        for apod in apod_options:
            # Set up CalcSpec_healpy object
            cs = CalcSpec_namaster([m0], apod, nside, bins, use_Dl=False, pure_B=True)
            # Calculate spectra
            spec = cs.calc([tqu])
            # Compare output spectra to the input power levels.
            fsky = np.mean(apod**2)**2 / np.mean(apod**4) # effective fsky for mask
            self.assertTrue(np.abs(Cl[0] - spec[0,0,0]) < TOL * np.sqrt(2) * Cl[0] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[1] - spec[1,0,0]) < TOL * np.sqrt(2) * Cl[1] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[2] - spec[2,0,0]) < TOL * np.sqrt(2) * Cl[2] / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(Cl[3] - spec[3,0,0]) < TOL * np.sqrt(Cl[0] * Cl[1] + Cl[3]**2) / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(spec[4,0,0]) < TOL * np.sqrt(Cl[1] * Cl[2]) / (2 * nside) / np.sqrt(fsky))
            self.assertTrue(np.abs(spec[5,0,0]) < TOL * np.sqrt(Cl[0] * Cl[2]) / (2 * nside) / np.sqrt(fsky))
        
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

class BpCovTest(unittest.TestCase):
    """
    Unit tests for bpcov.py

    """

    def test_mask_ell(self):
        """Test BpCov.mask_ell method"""

        # Make a BpCov structure with two maps, three ell bins.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        maplist = [map1, map2]
        bpcm = BpCov(maplist, 3)
        bpcm.set(np.ones((9,9)))

        # noffdiag=0
        self.assertTrue((bpcm.get(noffdiag=0) == np.array([[1,1,1,0,0,0,0,0,0],
                                                           [1,1,1,0,0,0,0,0,0],
                                                           [1,1,1,0,0,0,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,1,1,1,0,0,0],
                                                           [0,0,0,0,0,0,1,1,1],
                                                           [0,0,0,0,0,0,1,1,1],
                                                           [0,0,0,0,0,0,1,1,1]])).all())
        # noffdiag=1
        self.assertTrue((bpcm.get(noffdiag=1) == np.array([[1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,0,0,0],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [1,1,1,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1],
                                                           [0,0,0,1,1,1,1,1,1]])).all())
        # noffdiag=2
        self.assertTrue((bpcm.get(noffdiag=2) == np.ones((9,9))).all())
    
    def test_select_map(self):
        """Test BpCov.select method for maps"""
    
        # Make a BpCov structure with three maps, one ell bin.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        map3 = MapDef('map3', 'B')
        maplist = [map1, map2, map3]
        bpcm = BpCov(maplist, 1)
        bpcm.set(np.array([[ 0,  1,  2,  3,  4,  5],
                           [ 6,  7,  8,  9, 10, 11],
                           [12, 13, 14, 15, 16, 17],
                           [18, 19, 20, 21, 22, 23],
                           [24, 25, 26, 27, 28, 29],
                           [30, 31, 32, 33, 34, 35]]))
    
        # Select map1 only.
        bpcm1 = bpcm.select(maplist=[map1], ellind=None)
        self.assertTrue((bpcm1.get() == np.array([0])).all())
        # Select map2 only.
        bpcm2 = bpcm.select(maplist=[map2], ellind=None)
        self.assertTrue((bpcm2.get() == np.array([7])).all())
        # Select map3 only.
        bpcm3 = bpcm.select(maplist=[map3], ellind=None)
        self.assertTrue((bpcm3.get() == np.array([14])).all())
    
        # Select map1 and map2.
        bpcm12 = bpcm.select(maplist=[map1, map2], ellind=None)
        self.assertTrue((bpcm12.get() == np.array([[ 0,  1,  3],
                                                   [ 6,  7,  9],
                                                   [18, 19, 21]])).all())
        # Select map2 and map3
        bpcm23 = bpcm.select(maplist=[map2, map3], ellind=None)
        self.assertTrue((bpcm23.get() == np.array([[ 7,  8, 10],
                                                   [13, 14, 16],
                                                   [25, 26, 28]])).all())
        # Select map1 and map3
        bpcm13 = bpcm.select(maplist=[map1, map3], ellind=None)
        self.assertTrue((bpcm13.get() == np.array([[ 0,  2,  5],
                                                   [12, 14, 17],
                                                   [30, 32, 35]])).all())
    
        # Keep all three maps, but permute their order.
        bpcm231 = bpcm.select(maplist=[map2, map3, map1], ellind=None)
        self.assertTrue((bpcm231.get() == np.array([[ 7,  8,  6, 10, 11,  9],
                                                    [13, 14, 12, 16, 17, 15],
                                                    [ 1,  2,  0,  4,  5,  3],
                                                    [25, 26, 24, 28, 29, 27],
                                                    [31, 32, 30, 34, 35, 33],
                                                    [19, 20, 18, 22, 23, 21]])).all())

    def test_select_ell(self):
        """Test BpCov.select method for ell bins"""

        # Make a BpCov structure with two maps, three ell bins.
        map1 = MapDef('map1', 'B')
        map2 = MapDef('map2', 'B')
        maplist = [map1, map2]
        bpcm = BpCov(maplist, 3)
        M = np.ones((3,3))
        M = np.concat((M, 2*M, 4*M))
        M = np.concat((M, 3*M, 9*M), axis=1)
        bpcm.set(M)

        # Select bin 0 only.
        bpcm0 = bpcm.select(maplist=None, ellind=[0])
        self.assertTrue((bpcm0.get() == np.ones((3,3))).all())
        # Select bin 1 only.
        bpcm1 = bpcm.select(maplist=None, ellind=[1])
        self.assertTrue((bpcm1.get() == 6 * np.ones((3,3))).all())
        # Select bin 2 only.
        bpcm2 = bpcm.select(maplist=None, ellind=[2])
        self.assertTrue((bpcm2.get() == 36 * np.ones((3,3))).all())

        # Select bins 0 and 1.
        bpcm01 = bpcm.select(maplist=None, ellind=[0,1])
        self.assertTrue((bpcm01.get() == np.array([[1, 1, 1, 3, 3, 3],
                                                   [1, 1, 1, 3, 3, 3],
                                                   [1, 1, 1, 3, 3, 3],
                                                   [2, 2, 2, 6, 6, 6],
                                                   [2, 2, 2, 6, 6, 6],
                                                   [2, 2, 2, 6, 6, 6]])).all())
        # Select bins 1 and 2.
        bpcm12 = bpcm.select(maplist=None, ellind=[1,2])
        self.assertTrue((bpcm12.get() == np.array([[ 6,  6,  6, 18, 18, 18],
                                                   [ 6,  6,  6, 18, 18, 18],
                                                   [ 6,  6,  6, 18, 18, 18],
                                                   [12, 12, 12, 36, 36, 36],
                                                   [12, 12, 12, 36, 36, 36],
                                                   [12, 12, 12, 36, 36, 36]])).all())
        # Select bins 0 and 2.
        bpcm02 = bpcm.select(maplist=None, ellind=[0,2])
        self.assertTrue((bpcm02.get() == np.array([[1, 1, 1,  9,  9,  9],
                                                   [1, 1, 1,  9,  9,  9],
                                                   [1, 1, 1,  9,  9,  9],
                                                   [4, 4, 4, 36, 36, 36],
                                                   [4, 4, 4, 36, 36, 36],
                                                   [4, 4, 4, 36, 36, 36]])).all())
        # Select bins 2 and 0, i.e. flip the order.
        bpcm20 = bpcm.select(maplist=None, ellind=[2,0])
        self.assertTrue((bpcm20.get() == np.array([[36, 36, 36, 4, 4, 4],
                                                   [36, 36, 36, 4, 4, 4],
                                                   [36, 36, 36, 4, 4, 4],
                                                   [ 9,  9,  9, 1, 1, 1],
                                                   [ 9,  9,  9, 1, 1, 1],
                                                   [ 9,  9,  9, 1, 1, 1]])).all())
        
if __name__ == '__main__':
    unittest.main()



