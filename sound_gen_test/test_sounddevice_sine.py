#!/usr/bin/env python3
"""Play a sine signal."""
import argparse
import sys

import numpy as np
import sounddevice as sd


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


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
    '-d', '--device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-a', '--amplitude', type=float, default=0.2,
    help='amplitude (default: %(default)s)')
args = parser.parse_args(remaining)

start_idx = 0

try:
    samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
    #we try to query the classic sampelrate 4 our devie

    def callback(outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        global start_idx
        """
        In t we add the startindex to an arrange of the number of frames that
        is nparrray(1...1136), in that way we start at the start index until the
        end of our callback. Then we divided it by the sample rate so in that
        way we have our time stamp np arrang.
        """
        t = (start_idx + np.arange(frames)) / samplerate
        #print(t.shape)
        #In our case frames is equal to 1136
        """
        We reshape our time stamp we tranpose it 
        the nwe t will be (1136,1) (before 1136,)
        """
        t = t.reshape(-1, 1)
        #Outdata is the same shape as t
        #print(outdata)
        """
        for the first iteration we start with an np zeros of the shape of t
        then we rewrite data with our given sine array. After that they take the prevoious
        one and rewrite another. We make the operation to obtain our soind
        """

        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        start_idx += frames
        print(time.currentTime)


        """
        We add frames to the start index so in that way we can continue producing
        the smootheness of the sound
        Doc for time:
        https://python-sounddevice.readthedocs.io/en/0.3.3/_modules/sounddevice.html
        attributes currentTime
        """
        #print(start_idx)

    with sd.OutputStream(device=args.device, channels=1, callback=callback,
                         samplerate=samplerate):
        #Remember you can put many channels to make and add different sounds
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))