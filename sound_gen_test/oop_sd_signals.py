""" things we need: 
signals to change the output acordingly
a class with different attributes: SaberSound? Saber?
we need to have the previous state to generate the next one
by comparison, of velocities and changes in the courbe defining the laser
look if stationary. We need to define another class or all in one visualLaser who sends a signal to te other one
If we take into account the number of lasers it's when these two parallel lines are identified. At first we create 2 
instances of SaberLaser(color) and then if we define 2 // lines we add them to the SaberLaserColor class

"""
import time
import sys
from os import name
import sounddevice as sd
import argparse
import numpy as np
from scipy.signal import chirp, square, sawtooth

import threading
from multiprocessing import Lock

class SaberSound:
    def __init__(self, device, samplerate, amplitude):
        self.sample_rate = samplerate
        self.device = device
        self.amplitude = amplitude
        self.start_idx = 0

        self.data_mutex = Lock()
        self.value = 0 # Used inside the cb (i;e. the thread) only
        self.target_value = 0
        self.sound_gen = sd.OutputStream(device=self.device, channels=1, callback=self.on_sound,
                         samplerate= self.sample_rate)

    def start(self):
        self.sound_gen.__enter__()                # ugly ?
        
    def stop(self):
        self.sound_gen.__exit__(None, None, None) # ugly ?
            
    def on_sound(self, outdata, frames, time, status):
        with self.data_mutex:
            target_value = self.target_value
        alpha = .01
        self.value = alpha*target_value + (1-alpha)*self.value #  v += alpha(t-v)
        
        t = (self.start_idx + np.arange(frames)) / self.sample_rate
        noise = np.random.rand(frames)*.01*self.value
        cracks = np.zeros(frames)
        cracks[np.random.rand(frames) > .999] = 1 # 10% des valeurs à 1.
        freq = 200 #Hz
        signal = args.amplitude * (.1+self.value) * np.sin(2 * np.pi * freq * t) + cracks*.1 + noise

        # send signal to sound buffer
        outdata[:] = signal.reshape(-1, 1)
        self.start_idx += frames

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

    saber_sound = SaberSound(args.device, samplerate, args.amplitude)
    saber_sound.start()
    values = (1+np.sin(np.linspace(0, 10*np.pi, 100)))*.5
    for value in values:
        time.sleep(.1)
        saber_sound.set_value(value)
        value = not value
    saber_sound.stop()

    print('I am done')
        
        
    
    

        
