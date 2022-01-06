import time
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from multiprocessing import Lock

WAV_FILE_PATH = "sound_processing_FX/saber_sound.wav"


class SaberSound:
    def __init__(
        self,
        device=None,
        amplitude=0.2,
        wav_file_route=WAV_FILE_PATH,
    ):
        """
        A class used to simulate the sound of a lightsaber that change it's amplitude based on a value

        Attributes
        -----------
        device : Nonetype
            the device it should play the sound
        amplitude: float
            The amplitude of the signal, it values between 0 and 1
        wav_file_route: str


        """
        self.sample_rate_wav, self.sound_data = wavfile.read(wav_file_route)
        test_refactor = self.sound_data / np.linalg.norm(self.sound_data) * 100
        self.sound_data = test_refactor.reshape(-1, 1)
        self.device = device
        self.amplitude = amplitude
        self.parsed = False
        self.end_idx = 0
        self.data_mutex = Lock()
        self.value = 0  # Used inside the cb (i;e. the thread) only
        self.target_value = 0
        self.sound_gen = sd.OutputStream(
            device=self.device,
            channels=1,
            callback=self.on_sound,
            samplerate=self.sample_rate_wav,
        )

    def start(self):
        self.sound_gen.__enter__()  # ugly ?

    def stop(self):
        self.sound_gen.__exit__(None, None, None)  # ugly ?

    def set_end_idx(self, value):
        """
        Set the end index in a way that the wav file can be read in
        a loop.

        Parameters
        -----------
        value: int
            the end index we want to set initially

        """
        if value > len(self.sound_data):
            self.end_idx = value - len(self.sound_data)
            self.parsed = True
        else:
            self.end_idx = value
            self.parsed = False

    def on_sound(self, outdata, frames, _time_, _status_):
        """
        Gives the buffer the sound provided by the file and modifying it's
        amplitude based on the target value recieved in the thread.

        Parameters
        -----------
        outdata : np.array(1,1)
            output data we will listen to
        frames: int
            numer of frames that takes the chunk in the buffer

        Returns
        --------
        None

        """
        with self.data_mutex:
            target_value = self.target_value
        alpha = 0.01
        self.value = alpha * target_value + (1 - alpha) * self.value
        start_idx = self.end_idx
        self.set_end_idx(start_idx + frames)
        data_modified = (0.1 + self.value) * self.amplitude * self.sound_data
        if self.parsed:
            signal = np.concatenate(
                (data_modified[start_idx:], data_modified[: self.end_idx])
            )
        else:
            signal = data_modified[start_idx : self.end_idx]
        outdata[:] = signal

    def set_value(self, v):
        with self.data_mutex:
            self.target_value = v


if __name__ == "__main__":
    saber_sound = SaberSound()
    saber_sound.start()
    values = np.exp(np.linspace(-1, 1, 100))
    for value in values:
        time.sleep(0.1)
        saber_sound.set_value(value)

    saber_sound.stop()
    print("I am done")
