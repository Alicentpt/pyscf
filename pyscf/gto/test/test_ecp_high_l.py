# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regressions for high-angular-momentum off-center scalar ECP integrals."""

import ctypes
import unittest

import numpy
from scipy import special

from pyscf import gto, lib


class KnownValues(unittest.TestCase):
    """Independent radial-kernel and public-integral checks."""

    def test_bessel_large_z_high_order(self):
        """RF-01: Given order12; When z>16; Then match scaled SciPy values."""
        order = 12
        library = lib.load_library("libcgto")
        for name in ("ECPsph_ine", "ECPsph_ine_opt"):
            kernel = getattr(library, name)
            kernel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
            kernel.restype = None
            for z in (16.125, 32.0, 128.0):
                with self.subTest(kernel=name, z=z):
                    actual = numpy.zeros(order + 1)
                    kernel(actual.ctypes.data, order, z)
                    expected = numpy.sqrt(numpy.pi / (2 * z)) * special.ive(
                        numpy.arange(order + 1) + 0.5, z
                    )
                    numpy.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_off_center_high_l_coulomb(self):
        """RF-02: Given h/i AOs; When off-center ECP=-4/r; Then match libcint."""
        for angular_momentum in (5, 6):
            with self.subTest(angular_momentum=angular_momentum):
                mol = gto.M(
                    atom="Ce 0 0 0; Ne 0 0 1",
                    unit="Bohr",
                    basis={"Ce": [[angular_momentum, [1.0, 1.0]]]},
                    ecp={"Ne": gto.basis.parse_ecp("Ne nelec 10\nNe ul\n1 0.0 -4.0")},
                    charge=58,
                    spin=0,
                    verbose=0,
                )
                actual = mol.intor("ECPscalar")
                with mol.with_rinv_origin(mol.atom_coord(1)):
                    expected = -4 * mol.intor("int1e_rinv")
                numpy.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


if __name__ == "__main__":
    lib.num_threads(1)
    unittest.main()
