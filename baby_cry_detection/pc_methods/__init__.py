import librosa
import logging
import timeit

__all__ = [
    'Reader'
]


class Reader:
    """
    Read input audio file for training set
    file_name: 'path/to/file/filename.ogg'
    """

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Read audio file using pydub package. Pydub reads the file exactly as it is (no resampling, etc.)

        :return:
        * audio_data as numpy.ndarray. A two-dimensional NumPy array is returned, where the channels are stored
        along the first dimension, i.e. as columns. If the sound file has only one channel, a one-dimensional array is
        returned.
        * sr as int. The sample rate of the audio file [Hz]
        """

        logging.info('Reading file: {0} ...'.format(self.file_name))

        start = timeit.default_timer()

        audio_data, sr = librosa.load(self.file_name, sr=44100, mono=True, duration=5)

        stop = timeit.default_timer()

        logging.info('Time taken: {0}'.format(stop - start))

        return audio_data, sr
