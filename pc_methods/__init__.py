# -*- coding: utf-8 -*-

import soundfile as sf

__all__ = [
    'Reader'
]


class Reader:
    """
    Read input audio file
    file_name: 'path/to/file/filename.mp3'
    """

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Read audio file using read function of soundfile package

        :return:
        * audiodata as numpy.ndarray. A two-dimensional NumPy array is returned, where the channels are stored
        along the first dimension, i.e. as columns. If the sound file has only one channel, a one-dimensional array is
        returned. Use always_2d=True to return a two-dimensional array anyway.
        * samplerate as int. The sample rate of the audio file
        """

        return sf.read(self.file_name)
