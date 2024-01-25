import os
import numpy as np
import struct
import cv2


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)


def save(i, depth_map, norm_map, depth_save_path, normal_save_path):
    min_depth_percentile = 1.
    max_depth_percentile = 99.

    depth_map = read_array(depth_map)
    normal_map = read_array(norm_map)

    min_depth, max_depth = np.percentile(
        depth_map, [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    depth_map = cv2.normalize(depth_map, depth_map, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    depth_map = cv2.rotate(depth_map, 1)
    cv2.imwrite(depth_save_path + f'depth_{i}.tiff', depth_map)
    cv2.imwrite(normal_save_path + f'normal_{i}.tiff', normal_map)

    # import pylab as plt
    #
    # # Visualize the depth map.
    # plt.figure()
    # plt.imshow(depth_map)
    # plt.title("depth map")
    #
    # # Visualize the normal map.
    # plt.figure()
    # plt.imshow(normal_map)
    # plt.title("normal map")
    #
    # plt.show()


if __name__ == '__mia__':
    depth_folder = 'F:\\L7colmap\\stereo\\depth_maps\\'
    normal_folder = 'F:\\L7colmap\\stereo\\Normal_maps\\'
    depth_save_path = 'F:\\L7colmap\\stereo\\depth_images\\'
    normal_save_path = 'F:\\L7colmap\\stereo\\normal_images\\'

    depth_files = [depth_folder + _ for _ in os.listdir(depth_folder)]
    normal_files = [normal_folder + _ for _ in os.listdir(normal_folder)]
    i = 1
    for d, n in zip(depth_files, normal_files):
        save(i, d, n, depth_save_path, normal_save_path)
        i += 1
