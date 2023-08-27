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
        Read audio file using librosa package. librosa allows resampling to desired sample rate and convertion to mono.

        :return:
        * play_list: a list of audio_data as numpy.ndarray. There are 5 overlapping signals, each one is 5-second long.
        """

        play_list = list()

        for offset in range(5):
            audio_data, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
            play_list.append(audio_data)

        return play_list
