import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

import numpy as np
import random


class RoomSimulator():
    def __init__(self, sr=16000):
        # randomize variables
        self.room_x = random.random() * 2.5 + 3.0 # 3.0 ~ 5.5
        self.room_y = random.random() * 3 + 2.0 # 2.0 ~ 5.0
        self.room_z = random.random() * 1 + 2.0 # 2.0 ~ 3.0
        self.room_dim = [self.room_x, self.room_y, self.room_z]  # meters

        # use rt60
        rt60 = random.random() * 0.1 + 0.15 # 0.15 ~ 0.35
        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60, self.room_dim)
        material = pra.Material(e_absorption)
        self.sr = sr
        self.material = material
        self.max_order = 3
        print('room_dim', self.room_dim, 'RT60', rt60)

    def make_room(self):
        room = pra.ShoeBox(
            self.room_dim,
            fs=self.sr,
            materials=self.material,
            max_order=self.max_order,
            air_absorption=True,
            ray_tracing=True,
            use_rand_ism = True,
            max_rand_disp = 0.05
        )
        room.set_ray_tracing()
        
        return room

    def make_source_position(self):
        source1_x = self.room_x / 2
        source2_x = source1_x + random.random() * 0.5 + 0.9 # ~2.9
        source_y = self.room_y / 2

        mic_x = source1_x - 0.5 - random.random() * 0.75 # 0.25~
        mic_y = source_y + (random.random() - 0.5)
        return (source1_x, source_y, source2_x, source_y, mic_x, mic_y)
    
    def simulate(self, source1, source2, noise=None):
        (s1_x, s1_y, s2_x, s2_y, mic_x, mic_y) = self.make_source_position()
        s_z = 0.5

        # place the mic in the room
        mic_locs = np.c_[
            [mic_x, mic_y, s_z],  # mic 1
            # [source1_x, mic_y - 0.03, source_z],  # for 2ch mic
        ]

        if np.max(source1) > 1.0:
            source1 = source1 / np.max(source1)
        if np.max(source2) > 1.0:
            source2 = source2 / np.max(source2)

        # source1
        self.room = self.make_room()
        self.room.add_source([s1_x, s1_y, s_z], signal=source1, delay=0.0)
        self.room.add_microphone_array(mic_locs)
        self.room.simulate()
        source1 = self.room.mic_array.signals[0]
        
        # source2
        self.room = self.make_room()
        self.room.add_source([s2_x, s2_y, s_z], signal=source2, delay=0.0)
        self.room.add_microphone_array(mic_locs)
        self.room.simulate()
        source2 = self.room.mic_array.signals[0]

        return (source1, source2)
