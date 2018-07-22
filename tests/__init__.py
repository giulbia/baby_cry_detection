from unittest2 import TestCase
import librosa
import os


class TestBabyCry(TestCase):
    """
    Tests
    """

    @classmethod
    def setUpClass(cls):

        print "Set up test class"

        external_directory_path = '{}/../../external_input/'.format(os.path.dirname(os.path.abspath(__file__)))
        signal_name = 'signal_9s.ogg'

        cls.file_name = os.path.join(os.path.normpath(external_directory_path), signal_name)
        cls.pc_sample, _ = librosa.load(cls.file_name, sr=44100, mono=True, duration=5)
        cls.label = "test_label"

        cls.rpi_sample, _ = librosa.load(cls.file_name, sr=44100, mono=True, duration=9)
        cls.categories = ['301 - Crying baby', '901 - Silence', '902 - Noise', '903 - Baby laugh']

    @classmethod
    def tearDownClass(cls):

        print "\nTear down test class"
