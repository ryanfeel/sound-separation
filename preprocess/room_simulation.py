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
        self.room_x = random.random() * 3 + 2.5 # 2.5 ~ 5.5
        self.room_y = random.random() * 3 + 3.0 # 3.0 ~ 6.0
        self.room_z = random.random() + 2.0 + 0.2 # 2.2 ~ 3.2
        self.room_dim = [self.room_x, self.room_y, self.room_z]  # meters
        room_type = random.random() # 0 ~ 1.0
        print('room_dim', self.room_dim, ' room_type', room_type)
        
        if room_type > 0.01:
            # use rt60
            rt60 = random.random() * 0.3 + 0.15 # 0.15 ~ 0.45
            # We invert Sabine's formula to obtain the parameters for the ISM simulator
            e_absorption, max_order = pra.inverse_sabine(rt60, self.room_dim)
            material = pra.Material(e_absorption)
            print("RT60: ", rt60)
        else:
            # use material
            c_floor = random.choice(['carpet_soft_10mm', 'marble_floor', 'carpet_cotton', 'carpet_hairy'])
            c_wall = random.choice(['wooden_lining', 'plasterboard'])
            c_wall2 = random.choice(['glass_3mm', 'glass_window', "curtains_cotton_0.5", 
                'curtains_velvet', 'curtains_fabric', 'blinds_half_open', 'blinds_open'])
            print("materials: ", c_floor, c_wall, c_wall2)
            m = pra.make_materials(
                ceiling="plasterboard", # 석고보드
                floor=c_floor,
                east=c_wall,
                west=c_wall,
                north=c_wall,
                south=c_wall2
            )
            max_order = 10
            material = m

        self.sr = sr
        self.material = material
        self.max_order = max_order

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
        
        return room

    def make_source_position(self):
        source_case = []
        for x in range(3):
            if x == 2:
                source1_x = self.room_x - (random.random() * 0.2 + 0.3)
                source2_x = source1_x - (random.random() * 1.0 + 0.3)
                mic1_x = source1_x + random.random() * 0.2 + 0.1
                mic2_x = source2_x - random.random() * 0.2 - 0.1
            else:
                source1_x = self.room_x * x / 3 + random.random() * 0.2 + 0.3
                source2_x = source1_x + random.random() * 1.0 + 0.3
                mic1_x = source1_x - random.random() * 0.2 - 0.1
                mic2_x = source2_x + random.random() * 0.2 + 0.1
                if source2_x > self.room_x:
                    source2_x = self.room_x - 0.2
                    mic2_x = self.room_x - 0.01
            for y in range(2):
                if y == 0:
                    source_y = random.random() * 0.2 + 0.3
                    mic_y = random.random()
                else:
                    source_y = self.room_y - (random.random() * 0.2 + 0.3)
                    mic_y = self.room_y - random.random()
                # (s1_x, s1_y, s2_x, s2_y, mic_x, mic_y)
                source_case.append((source1_x, source_y, source2_x, source_y, mic1_x, mic_y))
                source_case.append((source2_x, source_y, source1_x, source_y, mic2_x, mic_y))
        for y in range(3):
            if y == 2:
                source1_y = self.room_y - (random.random() * 0.2 + 0.3)
                source2_y = source1_y - (random.random() * 1.0 + 0.3)
                mic1_y = source1_y + random.random() * 0.2 + 0.1
                mic2_y = source2_y - random.random() * 0.2 - 0.1
            else:
                source1_y = self.room_y * y / 3 + random.random() * 0.2 + 0.3
                source2_y = source1_y + random.random() * 1.0 + 0.3
                mic1_y = source1_y - random.random() * 0.2 - 0.1
                mic2_y = source2_y + random.random() * 0.2 + 0.1
                if source2_y > self.room_y:
                    source2_y = self.room_y - 0.2
                    mic2_y = self.room_y - 0.01
            for x in range(2):
                if x == 0:
                    source_x = random.random() * 0.2 + 0.3
                    mic_x = random.random()
                else:
                    source_x = self.room_x - (random.random() * 0.2 + 0.3)
                    mic_x = self.room_x - random.random()
                # (s1_x, s1_y, s2_x, s2_y, mic_x, mic_y)
                source_case.append((source_x, source1_y, source_x, source2_y, mic_x, mic1_y))
                source_case.append((source_x, source2_y, source_x, source1_y, mic_x, mic2_y))

        return random.choice(source_case)
    
    def simulate(self, source1, source2, noise=None):
        (s1_x, s1_y, s2_x, s2_y, mic_x, mic_y) = self.make_source_position()
        s_z = random.random() + 0.2

        # place the mic in the room
        mic_locs = np.c_[
            [mic_x, mic_y, s_z],  # mic 1
            # [source1_x, mic_y - 0.03, source_z],  # for 2ch mic
        ]

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

        # mixture
        self.room = self.make_room()
        self.room.add_source([s1_x, s1_y, s_z], signal=source1, delay=0.0)
        self.room.add_source([s2_x, s2_y, s_z], signal=source2, delay=0.0)
        self.room.add_microphone_array(mic_locs)
        self.room.simulate()
        mixture = self.room.mic_array.signals[0]

        return (source1, source2, mixture)
