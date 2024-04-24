from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from vlmaps.utils.clip_utils import get_text_feats_multiple_templates

#* pose, centers, contour, bbox
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d
from shapley.geometry import Point, Polygon

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.utils.clip_utils import get_lseg_score
from vlmaps.utils.navigation_utils import get_dist_to_bbox_2d #& by selsect_nearest_obj

#*vlmaps load
from vlmaps.utils.mapping_utils import load_3d_map


class VLMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        self.map_config = map_config
        # obstacles_path = os.path.join(map_dir, "obstacles.npy")
        # self.obstacles = load_map(obstacles_path)
        self.gs = map_config["grid_size"]
        self.cs = map_config["cell_size"]

        #*맵들 저장할 멤버 변수 정의
        self.mapped_iter_list = None
        self.grid_feat = None
        self.grid_pos = None
        self.weight = None
        self.occupied_ids = None
        self.grid_rgb = None

        self.obstacles_map = None
        self.obstacles_cropped = None

        self._setup_transforms()
        if data_dir:
            self._setup_paths(data_dir)
        # self.obstacles_new_cropped = None




        self.scores_mat = None
        self.categories = None

        # TODO: check if needed
        # map_path = os.path.join(map_dir, "grid_lseg_1.npy")
        # self.map = load_map(map_path)
        # self.map_cropped = self.map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        # self._init_clip()
        # self._customize_obstacle_map(
        #     map_config["potential_obstacle_names"],
        #     map_config["obstacle_names"],
        #     vis=False,
        # )
        # self.obstacles_new_cropped = Map._dilate_map(
        #     self.obstacles_new_cropped == 0,
        #     map_config["dilate_iter"],
        #     map_config["gaussian_sigma"],
        # )
        # self.obstacles_new_cropped = self.obstacles_new_cropped == 0
        # self.load_categories()
        # print("a VLMap is created")
        # pass





#?############################################################################################
#^ vlmaps load, data

    def load_map(self, data_dir: str) -> bool: #* h5df 형식의 vlmaps 데이터를 불러오는 함수
                                                #& LangRobot.load_scene_map에서 사용
        self._setup_paths(data_dir)
        self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
        if not self.map_save_path.exists():
            print("Loading VLMap failed because the file doesn't exist.")
            return False
        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
        ) = load_3d_map(self.map_save_path) #& import from vlmaps.utils.mapping_utils

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.semantic_dir = self.data_dir / "semantic"
        self.pose_path = self.data_dir / "poses.txt"
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
            self.semantic_paths = sorted(self.semantic_dir.glob("*.npy"))
        except FileNotFoundError as e:
            print(e)



#?############################################################################################
#^ pose, centers, contour, bbox

    #! 이해해야됨!!!!
    def get_nearest_pos(self, curr_pos: List[float], name: str) -> List[float]:
        contours, centers, bbox_list = self.get_pos(name) #* 지금 map_config.yaml에서 맵 타입이 vlmaps로 지정되어서 Map을 부모 클래스로 하는 VLMap을 사용하는 상태!, 따라서 get_pos는 VLMap에 정의되어있으므로 그걸 사용
        #* 해당 island에 해당되는 모든 island의 해당 정보 반환
        ids_list = self.filter_small_objects(bbox_list, area_thres=10) #* bbox가 지나치게 작은 경우 노이즈일 가능성이 높으므로 해당 island들은 필터링하기 위해 그 인덱스들의 리스트 획득
                                                                        #& map.py에 있던 함수
        contours = [contours[i] for i in ids_list] #* 이를 이용해 작은 친구들(노이즈)는 필터링
        centers = [centers[i] for i in ids_list]
        bbox_list = [bbox_list[i] for i in ids_list]
        if len(centers) == 0:
            return curr_pos #* 만약 해당 id에 해당되는 친구가 없다면 현재 위치를 반환
        id = self.select_nearest_obj(centers, bbox_list, curr_pos) #* 그렇지않다면 현재 위치에서 가장 가까운 친구의 인덱스를 획득

        return self.nearest_point_on_polygon(curr_pos, contours[id]) #* 해당 island의 경계선 중에서 현재 로봇 위치와 가장 가까운 지점의 좌표를 획득해 반환하는 것






    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]: #^ contours, cnenter, bbox_list를 뽑아 get_nearest_pos에 반환하는 함수
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories #! 이거 그 장애물 리스트? 같은건데 어디서 주어진 건지 못찾음 찾아봐야됨
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True) #* point cloud에 대한 mask
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs) #& import from vlmaps.utils.visualize_utils
                                                                        #* 3d로 되어있는 mask를 2d로 변환
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1] #* crop된 맵을 기준으로 mask를 잘라줌

        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        foreground = binary_closing(mask_2d, iterations=3) #& import from scipy.ndimage
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3) #& import from scipy.ndimage -> cv2에 들어가니깐 그런듯
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground) #& import from scipy.ndimage
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()
        #* ex. (1,1) 길이 4짜리 정사각형 -> contour: [−1.0,−1.0],[3.0,−1.0],[3.0,3.0],[−1.0,3.0] / bb : [-1,-1,3,3]
        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1) #* 1은 label_id인데, 장애물이 있는 위치를 나타내는게 1이니깐 이걸 넣어준 거 즉, 장애물이 있는 곳에 대한 아웃풋들을 구하겠다
        # print("centers", centers)                                                #& import from vlmaps.utils.index_utils - cv2 사용

        # whole map position
        #* crop된 맵을 기준으로 좌표를 구했으니 옆쪽의 잘린 길이들을 더해줘서 crop뙤지 않은 맵을 기준의 좌표로 변환해줌
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin
        #* 리턴되는 것은 그 id의 모든 island들의 contour, center, bbox의 리스트
        return contours, centers, bbox_list





    def index_map(self, language_desc: str, with_init_cat: bool = True): #* get_pos에서 사용되며, 해당 island에 해당되는 mask를 반환하는 함수
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories) #& import from vlmaps.utils.index_utils
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score( #& import from vlmaps.utils.clip_utils
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask
    



    def filter_small_objects(self, bbox_list: List[List[int]], area_thres: int = 50) -> List[int]:
        results_ids = []
        for bbox_i, bbox in enumerate(bbox_list):
            dx = bbox[1] - bbox[0]
            dy = bbox[3] - bbox[2]
            area = dx * dy
            if area > area_thres:
                results_ids.append(bbox_i)
        return results_ids



    def select_nearest_obj(
        self,
        centers: List[List[float]],
        bbox_list: List[List[float]],
        curr_pos: Tuple[float, float],
    ) -> int:
        dist_list = []
        for c, bbox in zip(centers, bbox_list):
            size = np.array([bbox[1] - bbox[0], bbox[3] - bbox[2]])
            dist = get_dist_to_bbox_2d(np.array(c), size, np.array(curr_pos)) #* 각 island 별 장애물과 로봇 사이 거리 구함
                                                                                #& import from vlmaps.utils.index_utils
            dist_list.append(dist) #* 이걸 리스트에 넣어줌
        id = np.argmin(dist_list) #* 그중에서 가장 작은 값을 갖는 녀석의 인덱스 번호를 획득
        return id #* 해당 인덱스 번호를 반환

    def nearest_point_on_polygon(self, coord: List[float], polygon: List[List[float]]):
        # Create a Shapely Point from the given coordinate
        point = Point(coord) #* shapely.geometry.Point 객체 생성 -> 기하 관련 패키지인듯
                            #& import from shapeley.geometry

        # Create a Shapely Polygon from the polygon's coordinates
        poly = Polygon(polygon) #* shapely.geometry.Polygon 객체 생성
                                #& import from shapeley.geometry

        # Find the nearest point on the polygon's boundary to the given point
        nearest = poly.exterior.interpolate(poly.exterior.project(point))

        # Extract the nearest point's coordinates as a tuple
        nearest_coords = [int(nearest.x), int(nearest.y)]

        return nearest_coords 