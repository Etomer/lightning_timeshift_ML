import h5py
import numpy as np
import pyroomacoustics as pra

def generate_paired_fft_dataset(
    target_path : str,
    n_examples : int = 2000,
    sample_length : int = 10000,
    reflection_coeff : float = 0.5,
    scatter_coeff : float = 0.15,
    speed_of_sound : float = 343.0, # m/s
    fs : float= 16000, # Hz
    max_freq : float = 4000, #Hz 
    room_min_size : float = 1.0, # meter
    room_max_size : float = 10.0, # meter
    ):
    raise NotImplementedError

def generate_impulse_response_dataset(
    target_path : str,
    n_rooms : int = 200,
    n_mics : int = 51,
    rir_len : int = 1600,
    reflection_coeff : float = 0.5,
    scatter_coeff : float = 0.15,
    speed_of_sound : float = 343.0, # m/s
    fs : float= 16000, # Hz
    room_min_size : float = 1.0, # meter
    room_max_size : float = 10.0, # meter
    ):
    speed_of_sound : float = 343.0
    metadata = locals()
    
    with h5py.File(target_path,"w") as hdf5_file:
        
        # write input of this function as metadata to the dataset
        for key in metadata:
            hdf5_file.attrs[key] = metadata[key]

        X = hdf5_file.create_dataset("input", (n_rooms,n_mics,rir_len), dtype="f")
        Y = hdf5_file.create_dataset("gt", (n_rooms,n_mics), dtype="f")
        
        for room_i in range(n_rooms):
            # randomly generate a rectangular cuboid
            x,y,z = (room_max_size - room_min_size)*np.random.rand(3) + room_min_size
            corners = np.array([[0,0], [0,y], [x,y], [x,0]]).T 
            room = pra.Room.from_corners(corners, fs=fs, max_order=2, materials=pra.Material(reflection_coeff, scatter_coeff), ray_tracing=True, air_absorption=True)
            room.extrude(z, materials=pra.Material(reflection_coeff, scatter_coeff))
            room.set_ray_tracing(receiver_radius=0.2, n_rays=10000, energy_thres=1e-5)

            #add sender and receivers to room
            random_point_in_room = lambda : np.random.rand(3)*[x,y,z]
            sender_position = random_point_in_room()
            room.add_source(sender_position)
            R = np.array(np.stack([random_point_in_room() for i in range(n_mics)]).T)
            room.add_microphone(R)
            
            # compute image sources for reflections
            room.image_source_model()
            room.compute_rir()

            for mic_i in range(n_mics):
                if len(room.rir[mic_i][0]) > rir_len:
                    X[room_i,mic_i] = room.rir[mic_i][0][:rir_len]
                    Y[room_i,mic_i] = np.linalg.norm(sender_position - R[:,mic_i])
                else:
                    X[room_i,mic_i,:len(room.rir[mic_i][0])] = room.rir[mic_i][0]
                    Y[room_i,mic_i] = np.linalg.norm(sender_position - R[:,mic_i])

def generate_moving_impulse_response_dataset(
    target_path : str,
    n_rooms : int = 200,
    n_mics : int = 51,
    rir_len : int = 1600,
    reflection_coeff : float = 0.5,
    scatter_coeff : float = 0.15,
    fs : float= 16000, # Hz
    room_min_size : float = 1.0, # meter
    room_max_size : float = 10.0, # meter
    sound_source_locations : int = 10,
    sound_source_max_move : float = 1.0, # meter
    directivity : bool = False, # makes speakers and microphones be directional
    ):
    """
    simulate dataset with moving sound source,

    Time note: Using default settings 1 room takes ~10 s to simulate and takes 3 MB of storage
    """
    speed_of_sound : float = 343.0
    metadata = locals()
    
    with h5py.File(target_path,"w") as hdf5_file:
        
        # write input of this function as metadata to the dataset
        for key in metadata:
            hdf5_file.attrs[key] = metadata[key]

        X = hdf5_file.create_dataset("input", (n_rooms, n_mics, sound_source_locations,rir_len), dtype="f")
        Y = hdf5_file.create_dataset("gt", (n_rooms,n_mics), dtype="f")
        
        for room_i in range(n_rooms):
            # randomly generate a rectangular cuboid
            x,y,z = (room_max_size - room_min_size)*np.random.rand(3) + room_min_size
            corners = np.array([[0,0], [0,y], [x,y], [x,0]]).T 
            if directivity:
                room = pra.ShoeBox([x,y,z], fs=fs, max_order=3, materials=pra.Material(reflection_coeff, scatter_coeff), ray_tracing=False, air_absorption=True)
            else:
                room = pra.Room.from_corners(corners, fs=fs, max_order=3, materials=pra.Material(reflection_coeff, scatter_coeff), ray_tracing=True, air_absorption=True)
                room.set_ray_tracing(receiver_radius=0.2, n_rays=10000, energy_thres=1e-5)
                room.extrude(z, materials=pra.Material(reflection_coeff, scatter_coeff))

            #add sender and receivers to room
            random_point_in_room = lambda : np.random.rand(3)*[x,y,z]
            sender_position_start = random_point_in_room()
            sender_position_end = random_point_in_room()
            
            
            if np.linalg.norm(sender_position_start - sender_position_end) > sound_source_max_move:
                sender_position_end = sender_position_start + sound_source_max_move*(sender_position_end - sender_position_start)/np.linalg.norm(sender_position_start - sender_position_end)
            
            
            for i in range(sound_source_locations):
                if directivity:
                    dir_obj = pra.directivities.CardioidFamily(
                    orientation=pra.directivities.DirectionVector(azimuth=np.random.rand()*360, colatitude=180*np.random.rand(), degrees=True),
                    pattern_enum=pra.directivities.DirectivityPattern.HYPERCARDIOID,
                    )
                    room.add_source(sender_position_end*i/(sound_source_locations-1) + sender_position_start*(sound_source_locations - i - 1)/(sound_source_locations-1),directivity=dir_obj)
                else:
                    room.add_source(sender_position_end*i/(sound_source_locations-1) + sender_position_start*(sound_source_locations - i - 1)/(sound_source_locations-1))
            
            sender_position_mid = (sender_position_start + sender_position_end)/2
            R = np.array(np.stack([random_point_in_room() for i in range(n_mics)]).T)
            if directivity:
                dir_objs = []
                for _ in range(n_mics):
                    dir_objs.append(pra.directivities.CardioidFamily(
                    orientation=pra.directivities.DirectionVector(azimuth=np.random.rand()*360, colatitude=180*np.random.rand(), degrees=True),
                    pattern_enum=pra.directivities.DirectivityPattern.HYPERCARDIOID,
                    ))
                room.add_microphone(R,directivity=dir_objs)
            else:
                room.add_microphone(R)

            
            # compute image sources for reflections
            room.image_source_model()
            room.compute_rir()

            for mic_i in range(n_mics):
                for sender_i in range(sound_source_locations):
                    if len(room.rir[mic_i][sender_i]) > rir_len:
                        X[room_i,mic_i, sender_i] = room.rir[mic_i][sender_i][:rir_len]
                        Y[room_i,mic_i] = np.linalg.norm(sender_position_mid - R[:,mic_i])
                    else:
                        X[room_i,mic_i, sender_i, :len(room.rir[mic_i][sender_i])] = room.rir[mic_i][sender_i]
                        Y[room_i,mic_i] = np.linalg.norm(sender_position_mid - R[:,mic_i])




