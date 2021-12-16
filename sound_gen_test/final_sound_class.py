
import time
import sys
from os import name
from scipy import signal
import sounddevice as sd
import argparse
import numpy as np
from scipy.io import wavfile

from multiprocessing import Lock

class SaberSound:
    def __init__(self, device, samplerate, amplitude,wav_file_route):
        self.sample_rate_wav, self.sound_data = wavfile.read(wav_file_route)
        test_refactor = self.sound_data/np.linalg.norm(self.sound_data)*100
        self.sound_data = test_refactor.reshape(-1,1)
        self.sample_rate = samplerate
        self.device = device
        self.amplitude = amplitude
        self.parsed = False
        self.end_idx= 0
        self.data_mutex = Lock()
        self.value = 0 # Used inside the cb (i;e. the thread) only
        self.target_value = 0
        self.sound_gen = sd.OutputStream(device=self.device, channels=1, callback=self.on_sound,
                         samplerate= self.sample_rate_wav)

    def start(self):
        self.sound_gen.__enter__()                # ugly ?
        
    def stop(self):
        self.sound_gen.__exit__(None, None, None) # ugly ?

    def set_end_idx(self, value):
        print(len(self.sound_data),value)
        if value > len(self.sound_data):
            self.end_idx = value - len(self.sound_data)
            self.parsed = True
        else:
            self.end_idx = value 
            self.parsed = False

            
    def on_sound(self, outdata, frames, time, status):
        with self.data_mutex:
            target_value = self.target_value
        alpha = .01
        self.value = alpha*target_value + (1-alpha)*self.value
        start_idx = self.end_idx
        self.set_end_idx(start_idx + frames)
        data_modified = (.1+self.value) *self.amplitude * self.sound_data
        if self.parsed:
            signal = np.concatenate((data_modified[start_idx:], data_modified[: self.end_idx]) )
            # signal[ : frames - self.end_idx ] = data_modified[start_idx:]
            # signal[frames - self.end_idx : ] = data_modified[: self.end_idx]
        else:
            signal = data_modified[start_idx: self.end_idx]
        # print(signal.shape)
        outdata[:] = signal
        

    def set_value(self, v):
        with self.data_mutex:
            self.target_value = v
                    
        


    

if __name__== "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'frequency', nargs='?', metavar='FREQUENCY', type=float, default=500,
        help='frequency in Hz (default: %(default)s)')
    parser.add_argument(
        '-d', '--device', type= int,
        help='output device (numeric ID or substring)')
    parser.add_argument(
        '-a', '--amplitude', type=float, default=0.2,
        help='amplitude (default: %(default)s)')
    args = parser.parse_args(remaining)

    samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
    wav_file = "./sound_gen_test/sound_analysis/good_loop_lightsaber.wav"
    wav_file2= "./sound_gen_test/sound_analysis/little_laser_sound.wav"

    saber_sound = SaberSound(args.device, samplerate, args.amplitude, wav_file)
    saber_sound.start()
    values = (1 + np.sin(np.linspace(0, 10*np.pi, 100)))*.5
    values2 = np.exp(np.linspace(-1,1,100))

    for value in values2:
        time.sleep(.1)
        saber_sound.set_value(value)
        value = not value
    saber_sound.stop()

    print('I am done')
        
        
    
    

        
