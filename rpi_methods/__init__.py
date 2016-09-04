# -*- coding: utf-8 -*-

import librosa

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
        Read audio file using librosa package. librosa allows re-sampling to desired sample rate and conversion to mono.

        :return:
        * audio_data as numpy.ndarray. A two-dimensional NumPy array is returned, where the channels are stored
        along the first dimension, i.e. as columns. If the sound file has only one channel, a one-dimensional array is
        returned.
        * sr as int. The sample rate of the audio file [Hz]
        """

        return librosa.load(self.file_name, sr=44100, mono=True, duration=5.0)
