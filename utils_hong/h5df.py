import h5py

# HDF5 파일 열기
file_path = '/home/hong/capstone/vlmaps/DATA/vlmaps_dataset/5LpN3gDmAk7_1/vlmap/vlmaps.h5df'
with h5py.File(file_path, 'r') as file:
    def printname(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print('Dataset')
            print('Shape:', obj.shape)
            print('Data type:', obj.dtype)
        elif isinstance(obj, h5py.Group):
            print('Group')
        print()

    # 파일 내 모든 객체 순회
    file.visititems(printname)