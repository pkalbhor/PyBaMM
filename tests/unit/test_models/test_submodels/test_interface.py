import pybamm
import unittest


class TestInterface(unittest.TestCase):
    def test_domain_failures(self):
        #         model = pybamm.interface.InterfacialReaction(None)

        #         with self.assertRaises(pybamm.DomainError):
        #             model.get_homogeneous_interfacial_current(None, "not a domain")

        #         with self.assertRaises(pybamm.DomainError):
        #             model.get_butler_volmer(None, None, "not a domain")

        #         with self.assertRaises(pybamm.DomainError):
        #             model.get_inverse_butler_volmer(None, None, "not a domain")

        #         c = pybamm.Variable("c", "not a domain")
        #         lithium_ion_model = pybamm.interface.LithiumIonReaction(None)
        with self.assertRaises(pybamm.DomainError):
            pybamm.interface.BaseInterface(None, "not a domain")


class TestInterfaceLeadAcid(unittest.TestCase):
    def test_domain_failures(self):
        c = pybamm.Variable("c", "not a domain")
        lead_acid_model = pybamm.interface_lead_acid.MainReaction(None)
        with self.assertRaises(pybamm.DomainError):
            lead_acid_model.get_exchange_current_densities(c, "not a domain")


class TestOxygenReaction(unittest.TestCase):
    def test_butler_volmer_failure(self):
        interface = pybamm.interface_lead_acid.OxygenReaction(None)
        with self.assertRaises(ValueError):
            interface.get_butler_volmer(None, None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
