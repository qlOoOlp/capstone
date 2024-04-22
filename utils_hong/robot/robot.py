import os
import pickle
import numpy as np
import h5py
from omegaconf import DictConfig, OmegaConf

from utils_hong.robot.map import Map
#from vlmaps.map.map import Map

from typing import List, Tuple, Dict, Any


class LangRobot:
    """
    This class provides all primitives API that the robot can call during navigation
    """

    def __init__(self): #, config: DictConfig):
#        self.config = config
        self.vlmaps_data_save_dirs= ????
        self.curr_pos_on_map = None
        self.curr_ang_deg_on_map = None
        pass





    #* object_goal_navigation에서 사용됨! 목표지점으로 이동시키는 함수
    def move_to_object(self, name: str):
        self._set_nav_curr_pose() #* 일단 현재 위치 정보 최신화 하고
        pos = self.map.get_nearest_pos(self.curr_pos_on_map, name) #* 해당 name의 객체 중 로봇과 가장 가까운 객체에서도 로봇 위치와 가장 가까운 지점을 획득해 반환
        self.move_to(pos)




    def _set_nav_curr_pose(self): #! habitat_lang_robot에서 가져온 함수
        """
        Set self.curr_pos_on_map and self.curr_ang_deg_on_map
        based on the simulator agent ground truth pose
        """
        agent_state = self.sim.get_agent(0).get_state() #! 이거 habitat임 쓰면 안됨
        hab_tf = agent_state2tf(agent_state) #! 이것도 habitat네
        self.vlmaps_dataloader.from_habitat_tf(hab_tf)
        row, col, angle_deg = self.vlmaps_dataloader.to_full_map_pose()
        self.curr_pos_on_map = (row, col)
        self.curr_ang_deg_on_map = angle_deg
        print("set curr pose: ", row, col, angle_deg)


    def get_vlmaps_data_dirs(self):
        return self.vlmaps_data_save_dirs

    #! 여기서 scene information에 대한 config는 제거하고 그냥 데이터 직접 불러오게 바꾸고 싶음 일단 테스트니깐 / 나중에 리펙토링 할 때나 이렇게 쓰고
    def load_scene_map(self, data_dir: str, map_config: DictConfig):
        self.map = Map.create(map_config)
        self.map.load_map(data_dir)
        self.map.generate_obstacle_map() #* 옵스타클 맵이 demo에선 VLMaps랑 같이 만들어졌는데, master에선 VLMaps 이용해서 만들어지네



    #! habitat_lang_robot에서 가져온 함수
    def setup_map(self, vlmaps_data_dir: str):
        self.load_scene_map(vlmaps_data_dir, self.config["map_config"])

        # TODO: check if needed
        if "3d" in self.config.map_config.map_type:
            self.map.init_categories(mp3dcat.copy())
            self.global_pc = grid_id2base_pos_3d_batch(self.map.grid_pos, self.cs, self.gs)

        self.vlmaps_dataloader = VLMapsDataloaderHabitat(vlmaps_data_dir, self.config.map_config, map=self.map)





    def empty_recorded_actions(self):
        self.recorded_actions_list = [] #* action들이 저장될 리스트
        self.recorded_robot_pos = [] #* robot의 위치가 저장될 리스트
        self.goal_tfs = None #* ???
        self.all_goal_tfs = None #* ???
        self.goal_id = None #* ???
``
    #* 이거 함수 쓰기 전에 반드시 empty_recorded_actions() 함수를 먼저 호출했었는지 확인해야 recorded_actions_list가 호출될 수 있음
    def get_recorded_actions(self):
        return self.recorded_actions_list
    
#?############################################################################################

    def load_code(self, code_dir: str, task_i: int):
        code_path = os.path.join(code_dir, f"{task_i:06}.txt")
        with open(code_path, "r") as f:
            code = f.readlines()
        code = "".join(code)
        return code

    def execute_actions(self, actions_list: List[Any]):
        return NotImplementedError

    def _execute_action(self, action: str):
        return NotImplementedError

    def get_sound_pos(self, name: str):
        return NotImplementedError
        # tfs = self.sound_map.get_pos(name)

    def get_agent_pose_on_map(self) -> Tuple[float, float, float]:
        """
        Return row, col, angle_degree on the full map
        """
        return (
            self.curr_pos_on_map[0],
            self.curr_pos_on_map[1],
            self.curr_ang_deg_on_map,
        )

    #* map에 정의되어있는 동일 이름의 함수 사용해서 뭐 어떻게어떻게 꼼지락대는 것 같은데 흠 잘모르겠다 왜 있어야되는지는
    def get_pos(self, name: str):
        """
        Return nearest object position on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            return self.curr_pos_on_map
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        center = centers[nearest_id]
        return center

    def get_contour(self, name: str) -> List[List[float]]:
        """
        Return nearest object contour points on the map
        """
        contours, centers, bbox_list = self.map.get_pos(name)
        if not centers:
            print(f"no objects {name} detected")
            assert False
        ids = self.map.filter_small_objects(bbox_list)
        if ids:
            centers = [centers[x] for x in ids]
            bbox_list = [bbox_list[x] for x in ids]
            contours = [contours[x] for x in ids]

        nearest_id = self.map.select_nearest_obj(centers, bbox_list, self.curr_pos_on_map)
        contour = contours[nearest_id]
        return contour



#?############################################################################################







    def move_to(self, pos: Tuple[float, float]):
        """
        Move the robot to the position on the map
        based on accurate localization in the environment
        """
        # check if the pos is None
        return NotImplementedError

    def turn(self, angle_deg: float):
        return NotImplementedError
        # actions_list = self.nav.turn(angle_deg)
        # self.execute_actions(actions_list)
        # self.actions_list += actions_list


    #& move_to의 habitat_sim 버전
    # def move_to(self, pos: Tuple[float, float]) -> List[str]:
    #     """Move the robot to the position on the full map
    #         based on accurate localization in the environment

    #     Args:
    #         pos (Tuple[float, float]): (row, col) on full map

    #     Returns:
    #         List[str]: list of actions
    #     """
    #     actual_actions_list = []
    #     success = False
    #     # while not success:
    #     self._set_nav_curr_pose()
    #     curr_pose_on_full_map = self.get_agent_pose_on_map()  # (row, col, angle_deg) on full map
    #     paths = self.nav.plan_to(
    #         curr_pose_on_full_map[:2], pos, vis=self.config["nav"]["vis"]
    #     )  # take (row, col) in full map
    #     print(paths)
    #     actions_list, poses_list = self.controller.convert_paths_to_actions(curr_pose_on_full_map, paths[1:])
    #     success, real_actions_list = self.execute_actions(actions_list, poses_list, vis=self.config["nav"]["vis"])
    #     actual_actions_list.extend(real_actions_list)

    #     actual_actions_list.append("stop")

    #     if not hasattr(self, "recorded_actions_list"):
    #         self.recorded_actions_list = []
    #     self.recorded_actions_list.extend(actual_actions_list)

    #     return actual_actions_list


    #& turn의 habitat_sim 버전
    # def turn(self, angle_deg: float):
    #     """
    #     Turn right a relative angle in degrees
    #     """
    #     if angle_deg < 0:
    #         actions_list = ["turn_left"] * int(np.abs(angle_deg / self.turn_angle))
    #     else:
    #         actions_list = ["turn_right"] * int(angle_deg / self.turn_angle)

    #     success, real_actions_list = self.execute_actions(actions_list, vis=self.config.nav.vis)

    #     self.recorded_actions_list.extend(real_actions_list)
    #     return real_actions_list



#?############################################################################################

    #* LLM 연결에서 사용되는 이동 관련 함수
    def with_object_on_left(self, name: str):
        self.face(name)
        self.turn(90)

    def with_object_on_right(self, name: str):
        self.face(name)
        self.turn(-90)

    def move_to_left(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_left_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_to_right(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_right_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        #print("\n\nsdfsdafasfh sadvsdfsfsfsdfsdf",pos)
        self.move_to(pos)

    def move_in_between(self, name_a: str, name_b: str):
        self._set_nav_curr_pose()
        pos = self.map.get_pos_in_between(self.curr_pos_on_map, self.curr_ang_deg_on_map, name_a, name_b)
        self.move_to(pos)

    def turn_absolute(self, angle_deg: float):
        self._set_nav_curr_pose()
        delta_deg = angle_deg - self.curr_ang_deg_on_map
        actions_list = self.turn(delta_deg)
        self.recorded_actions_list.extend(actions_list)

    def face(self, name: str):
        self._set_nav_curr_pose()
        turn_right_angle = self.map.get_delta_angle_to(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.turn(turn_right_angle)

    def move_north(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_north_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_south(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_south_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_west(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_west_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_east(self, name: str):
        self._set_nav_curr_pose()
        pos = self.map.get_east_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, name)
        self.move_to(pos)

    def move_forward(self, meters: float):
        self._set_nav_curr_pose()
        pos = self.map.get_forward_pos(self.curr_pos_on_map, self.curr_ang_deg_on_map, meters)
        self.move_to(pos)