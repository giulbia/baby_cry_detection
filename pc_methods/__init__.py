import pydub
import numpy as np

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

        # Create a silent sound of exactly 5 seconds
        silent_template = pydub.AudioSegment.silent(duration=5000)

        # Read file
        sound = pydub.AudioSegment.from_file(self.file_name)

        # Trim 5 seconds
        sound_5s = silent_template.overlay(sound[0:5000])

        # Convert to AudioSegment object to array. /!\ should it be float32 ??? /!\
        audio_data = (np.fromstring(sound_5s._data, dtype="int16") + 0.00000000001) / (0x7FFF + 0.00000000001)

        # Sample rate
        sr = sound.frame_rate

        return audio_data, sr
