from tests import TestBabyCry


class PcMethodsTest(TestBabyCry):
    """
    Test pc_methods
    """

    def test_ciao(self):
        self.assertEqual(1, 1)

    def test_ciao2(self):
        self.assertFalse(1 < 0)
