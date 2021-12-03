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

class SaberSound:
    def __init__(self, device, samplerate, amplitude) -> None:
        self.sample_rate = samplerate
        self.device = device
        self.amplitude = amplitude
        self.start_idx = 0
        self.value = 0
        self.sound_gen = sd.OutputStream(device=self.device, channels=1, callback=self.on_sound,
                         samplerate= self.sample_rate)


    def on_sound(self, outdata, frames, time, status):
        t = (self.start_idx + np.arange(frames)) / self.sample_rate
        t = t.reshape(-1, 1)
        if self.value ==0:
            outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        elif self.value==1:
            outdata[:] = args.amplitude * square(2 * np.pi * args.frequency * t)
        self.start_idx += frames

    def set_value(self):
        if self.value ==1:
            self.value = 0
        else:
            self.value = 1

    def loop(self):
        with self.sound_gen:
            while True:
                self.set_value()
                time.sleep(0.4)
                print(self.value)
                    
        


    

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

    Test1 = SaberSound(args.device, samplerate, args.amplitude)
    Test1.loop()
    #Test1

        
