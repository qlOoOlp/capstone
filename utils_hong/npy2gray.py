import sys
import os
from PIL import Image
import numpy as np

def check_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def convert_npy2gray(input_dir,output_dir):
    file_list = os.listdir(input_dir)
    npy_files = [f for f in file_list if f.endswith('.npy')]
    for npy in npy_files:
        npy_path = os.path.join(input_dir, npy)
        img = _convert_npy2gray(npy_path)
        gray_file = os.path.splitext(npy)[0] + '.png'
        gray_path = os.path.join(output_dir, gray_file)
        image = Image.fromarray(img.astype('uint8'), 'L')  # 'L'은 그레이스케일을 의미
        image.save(gray_path)
    return hi
        
def _convert_npy2gray(npy_path):
    img = np.load(npy_path)
    # 데이터의 최소값과 최대값을 이용한 정규화
    #print(img)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val != min_val:  # 모든 값이 같지 않을 경우
        normalized_depth = (img - min_val) / (max_val - min_val) * 255
    else:
        normalized_depth = img * 0  # 모든 값이 같을 경우 검은 이미지 생성

    return normalized_depth.astype(np.uint8)


if __name__ == "__main__":
    in_dir = "/home/hong/capstone/vlmaps/data_demo/hong2/depth"
    out_dir = "/home/hong/capstone/vlmaps/data_demo/hong2/depth_img"
    check_dir(out_dir)
    convert_npy2gray(in_dir, out_dir)