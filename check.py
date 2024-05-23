def copy_obj_to_txt(src_filename, dest_filename):
    # 파일을 읽고 그대로 다른 파일에 쓰기
    with open(src_filename, 'r') as src_file:
        with open(dest_filename, 'w') as dest_file:
            for line in src_file:
                dest_file.write(line)

# 사용 예
src_filename = '/home/hong/Desktop/240513-181453.obj'
dest_filename = 'output_text_file.txt'
copy_obj_to_txt(src_filename, dest_filename)