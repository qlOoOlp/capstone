import sys
import os
from PIL import Image
import numpy as np

# 변환할 PNG 파일이 있는 디렉토리 경로
input_directory = "/path/to/input_directory"

# 변환된 NPY 파일을 저장할 디렉토리 경로
output_directory = "/path/to/output_directory"

def convert_png_to_npy(input_dir, output_dir):
# 입력 디렉토리 내의 모든 파일 목록 가져오기
    file_list = os.listdir(input_dir)
    
    # PNG 파일만 선택하여 처리
    png_files = [f for f in file_list if f.endswith('.png')]
    
    for png_file in png_files:
        # PNG 파일의 경로
        png_path = os.path.join(input_dir, png_file)
        img_array = _convert_png_to_npy(png_path)

        # NPY 파일의 경로
        npy_file = os.path.splitext(png_file)[0] + '.npy'
        npy_path = os.path.join(output_dir, npy_file)
        
        # NumPy 배열을 NPY 파일로 저장
        np.save(npy_path, img_array)
        
        print(f"Converted {png_file} to {npy_file}")

def _convert_png_to_npy(png_path):
    
    # PNG 파일 열기
    img = Image.open(png_path)
    
    # 이미지를 NumPy 배열로 변환
    return np.array(img)
        


if __name__ == "__main__":
    # 입력 및 출력 디렉토리 경로를 가져오기
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    convert_png_to_npy(input_directory, output_directory)
