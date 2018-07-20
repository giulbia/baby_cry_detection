from unittest2 import TestCase
import numpy as np


class TestBabyCry(TestCase):
    """
    Tests
    """

    @classmethod
    def setUpClass(cls):

        print "Set up test class"
        cls.sample = np.ndarray([3, 2, 1])

    @classmethod
    def tearDownClass(cls):

        print "Tear down test class"
