import os, sys, h5py
sys.path.append(os.getcwd())
from src.datasets import generate_moving_impulse_response_dataset
from multiprocessing import Pool


# Setiing ----------------------------
dataset_name = "moving_dataset_directivity.hdf5" # with .hdf5 at the end
target_path = os.path.join(".","data","datasets", dataset_name)
n_cores = 8
# specify arguments in the inner function bellow
def generator(target_path_part):
    generate_moving_impulse_response_dataset(target_path_part, n_rooms=1500, directivity=True,reflection_coeff=0.1)
#generator = lambda target_path_part : generate_moving_impulse_response_dataset(target_path_part, n_rooms=1)
# -------------------------------


if __name__ == "__main__":
    if n_cores == 1:
        generator(target_path)
    else:
        piece_paths = [target_path[:-5] + "_piece_" + str(i) + ".hdf5" for i in range(n_cores)]
        with Pool(n_cores) as p:
            p.map(generator, piece_paths)

        # combine pieces to final dataset
        with h5py.File(target_path, "w") as hdf5_file:
            with h5py.File(piece_paths[0], "r") as hdf5_piece:
                piece_dataset_size = hdf5_piece["input"].shape[0]
                X = hdf5_file.create_dataset(
                    "input", [dim_size if i != 0 else n_cores*dim_size for i,dim_size in enumerate(hdf5_piece["input"].shape)], dtype="f"
                )
                Y = hdf5_file.create_dataset("gt", [dim_size if i != 0 else n_cores*dim_size for i,dim_size in enumerate(hdf5_piece["gt"].shape)], dtype="f")
                for key in hdf5_piece.attrs.keys():
                    hdf5_file.attrs[key] = hdf5_piece.attrs[key]
                hdf5_file.attrs["n_rooms"] = hdf5_piece.attrs["n_rooms"]*n_cores

            for i in range(n_cores):
                with h5py.File(piece_paths[i], "r") as hdf5_read_file:
                    X[i * piece_dataset_size : (i + 1) * piece_dataset_size] = hdf5_read_file["input"]
                    Y[i * piece_dataset_size : (i + 1) * piece_dataset_size] = hdf5_read_file["gt"]
        for piece_path in piece_paths:
            os.remove(piece_path)
