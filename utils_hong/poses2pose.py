import sys
import os

def check_dir(pose_dir):
    if not os.path.exists(pose_dir):
        os.makedirs(pose_dir)
def poses2pose(poses_file, pose_dir):
    # poses.txt 파일에서 pose 정보 읽기
    with open(poses_file, "r") as file:
        poses = file.readlines()
    # pose 정보를 각각의 파일에 저장
    for idx, pose_info in enumerate(poses[1:]):
        pose_info = pose_info.split(' ')[1:]
        pose_info = '\t'.join(pose_info)

        # pose 파일명 생성 (000000.txt, 000001.txt, ...)
        pose_filename = f"{idx:06}.txt"
        # pose 파일 경로 설정
        pose_filepath = os.path.join(pose_dir, pose_filename)
        # pose 정보를 파일에 쓰기
        with open(pose_filepath, "w") as pose_file:
            pose_file.write(pose_info)
    print(f"poses.txt -> {len(poses[1:])} pose.txt done")
if __name__ =="__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <poses_file> <pose_dir>")
        sys.exit(1)
    poses_file=sys.argv[1]
    pose_dir=sys.argv[2]
    check_dir(pose_dir)
    poses2pose(poses_file, pose_dir)