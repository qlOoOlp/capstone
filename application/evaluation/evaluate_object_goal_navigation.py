import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra

from utils_hong.robot.robot import LangRobot
#from vlmaps.task.habitat_object_nav_task import HabitatObjectNavigationTask
#from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.matterport3d_categories import mp3dcat

#^ map config가 vlmap으로 되어있으므로 map관련해서는 Map을 부모 클래스로 하는 VLMap을 봐야됨

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = HabitatLanguageRobot(config)
    object_nav_task = HabitatObjectNavigationTask(config)
    object_nav_task.reset_metrics()
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())
        object_nav_task.setup_scene(robot.vlmaps_dataloader)
        object_nav_task.load_task() #* self.task_dict에 task들이 들어감
        # print("\n\n\nstep1")
        # print(object_nav_task.task_dict)
        """
        ^ object_nav_task. task_dict의 구조
&         [{'task_id': 0, 'scene': '5LpN3gDmAk7_1', 'map_grid_size': 1000, 'map_cell_size': 0.05, 'tf_habitat': [[1.0, 0.0, 0.0, 8.37611164209252], [0.0, 1.0, 0.0, -1.2843499183654785], [0.0, 0.0, 1.0, 7.724077207991911], [0.0, 0.0, 0.0, 1.0 
&        ]], 'instruction': 'Go to the closest cushion first, then go to a chair nearby, after that, go to a counter and in the end, navigate to a table.', 'objects_info': [{'name': 'cushion', 'row': 623, 'col': 413}, {'name': 'chair', 'row 
&        ': 607, 'col': 438}, {'name': 'counter', 'row': 733, 'col': 507}, {'name': 'table', 'row': 720, 'col': 561}]}, {'task_id': 1, 'scene': '5LpN3gDmAk7_1', 'map_grid_size': 1000, 'map_cell_size': 0.05, 'tf_habitat': [[1.0, 0.0, 0.0, 2. 
&        910856243701909], [0.0, 1.0, 0.0, -1.2843499183654785], [0.0, 0.0, 1.0, 4.726257791153177], [0.0, 0.0, 0.0, 1.0]], 'instruction': 'First approach the stairs, then find a nearby sofa and go there, next come to a picture before final 
&        ly navigate to a sink.', 'objects_info': [{'name': 'stairs', 'row': 683, 'col': 290}, {'name': 'sofa', 'row': 724, 'col': 358}, {'name': 'picture', 'row': 767, 'col': 304}, {'name': 'sink', 'row': 791, 'col': 281}]}, 



&        {'task_id': 2, 
&        'scene': '5LpN3gDmAk7_1', 'map_grid_size': 1000, 'map_cell_size': 0.05, 'tf_habitat': [[1.0, 0.0, 0.0, 12.918961066071216], [0.0, 1.0, 0.0, -1.2843499183654785], [0.0, 0.0, 1.0, -0.024132977068202874], [0.0, 0.0, 0.0, 1.0]], 'inst 
&        ruction': 'Turn around and find a chair, go to a table in the front and then walk to the counter, finally move to the sofa.', 'objects_info': [{'name': 'chair', 'row': 544, 'col': 508}, {'name': 'table', 'row': 719, 'col': 558}, {' 
&         name': 'counter', 'row': 749, 'col': 505}, {'name': 'sofa', 'row': 752, 'col': 554}]}, 
        """
        
        
        
        for task_id in range(len(object_nav_task.task_dict)):
            object_nav_task.setup_task(task_id) #* 이렇게 하면 앞서 .task_dict에 있던 멤버들이 object_nav_task 본인의 멤버로 가져와짐!!!!
            # print("\n\n\nstep2")
            # print(object_nav_task.task_dict[task_id])
            # print("\n\n\nstep3")
            # print(object_nav_task.instruction)
            """
&            │step2                                                                                                                                                                                    │
&            │{'task_id': 0, 'scene': '5LpN3gDmAk7_1', 'map_grid_size': 1000, 'map_cell_size': 0.05, 'tf_habitat': [[1.0, 0.0, 0.0, 8.37611164209252], [0.0, 1.0, 0.0, -1.2843499183654785], [0.0, 0.0,│
&            │ 1.0, 7.724077207991911], [0.0, 0.0, 0.0, 1.0]], 'instruction': 'Go to the closest cushion first, then go to a chair nearby, after that, go to a counter and in the end, navigate to a ta│
&            │ble.', 'objects_info': [{'name': 'cushion', 'row': 623, 'col': 413}, {'name': 'chair', 'row': 607, 'col': 438}, {'name': 'counter', 'row': 733, 'col': 507}, {'name': 'table', 'row': 720│
&            │, 'col': 561}]}                                                                                                                                                   

&            │step3                                                                                                                                                                                    │
&            │Go to the closest cushion first, then go to a chair nearby, after that, go to a counter and in the end, navigate to a table.                   
            """
        
            object_categories = parse_object_goal_instruction(object_nav_task.instruction) #* 여기서 LLM 적용된다, object_nav_task.instaruction은 for문 스텝에 따른 task_id에서의 이동 명령 자연어 ex. (Go to the closest cushion first, then go to a chair nearby, after that, go to a counter and in the end, navigate to a table. )
            print(f"instruction: {object_nav_task.instruction}")
            robot.empty_recorded_actions() #* LangRobot class에 있으며, 이동 명령 획득을 위한 초기 세팅 담당
            """
            * 초기화되는 것들
            * def empty_recorded_actions(self):
            * self.recorded_actions_list = [] #* action들이 저장될 리스트
            * self.recorded_robot_pos = [] #* robot의 위치가 저장될 리스트
            * self.goal_tfs = None #* goal의 변환행렬??
            * self.all_goal_tfs = None #* 모든 goal의 변환행렬을 저장해둔 리스트?
            * self.goal_id = None #* 목표지점의 id
            
            """
            robot.set_agent_state(object_nav_task.init_hab_tf) #* HabitatLanguageRobot class에 있으며, 로봇의 위치를 설정해주는 함수이며, 여기선 초기 위치를 넣었기에 초기위치 세팅
                                                                #* init_hab_tf는 task_dict의 첫번째 task의 tf_habitat이며, 이는 habitat simulator의 변환행렬을 의미하는 듯
            for cat_i, cat in enumerate(object_categories): #* object_categories는 목표 키워드의 리스트이므로 cat_i에는 목표 키워드의 인덱스 번호, cat에는 해당 스텝에서의 목표 키워드가 들어감
                print(f"Navigating to category {cat}")
                actions_list = robot.move_to_object(cat) #* 이게 target 키워드로 이동할 때의 이동 명령의 리스트를 반환해주는 함수인듯
            recorded_actions_list = robot.get_recorded_actions()
            robot.set_agent_state(object_nav_task.init_hab_tf)
            for action in recorded_actions_list:
                object_nav_task.test_step(robot.sim, action, vis=config.nav.vis)

            save_dir = robot.vlmaps_dataloader.data_dir / (config.map_config.map_type + "_obj_nav_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / f"{task_id:02}.json"
            object_nav_task.save_single_task_metric(save_path)


if __name__ == "__main__":
    main()



#?############################################################################################
from vlmaps.robot.lang_robot import LangRobot

def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = LangRobot(config)
#    object_nav_task = HabitatObjectNavigationTask(config)
#    object_nav_task.reset_metrics()
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())
        object_nav_task.setup_scene(robot.vlmaps_dataloader)


#! 일단 scene number 여러개 안하고 하나만 하도록 해보자###############################################################################

from vlmaps.robot.lang_robot import LangRobot
from vlmaps.utils.matterport3d_categories import mp3dcat

def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = LangRobot(config)
#    object_nav_task = HabitatObjectNavigationTask(config)
#    object_nav_task.reset_metrics()
#    scene_ids = []
#    if isinstance(config.scene_id, int):
#        scene_ids.append(config.scene_id)
#    else:
#        scene_ids = config.scene_id
    robot.scene_id=scene_id
    vlmaps_data_dir=robot.vlmaps_data_save_dirs[scene_id]
    print(vlmaps_data_dir)
    robot.scene_name=vlmaps_data_dir.name.split("_")[0]
    robot._setup_sim(robot.scene_name)
    robot.setup_map(vlmaps_data_dir)

    cropped_obst_map = robot.map.get_obstacle_cropped()
    if robot.config.map_config.potential_obstacle_names and robot.config.map_config.obstacle_names:
        print("come here")
        robot.map.customize_obstacle_map(
            robot.config.map_config.potential_obstacle_names,
            robot.config.map_config.obstacle_names,
            vis=robot.config.nav.vis,
        )
        cropped_obst_map = robot.map.get_customized_obstacle_cropped()
    
    robot.nav.build_visgraph(
        cropped_obst_map,
        robot.vlmaps_dataloader.rmin,
        robot.vlmaps_dataloader.cmin,
        vis=robot.config["nav"]["vis"],
    )


    robot.map.init_categories(mp3dcat.copy())
    object_nav_task.setup_scene(robot.vlmaps_dataloader)

    for task_id in range(len(object_nav_task.task_dict)):
        object_nav_task.setup_task(task_id) #* 이렇게 하면 앞서 .task_dict에 있던 멤버들이 object_nav_task 본인의 멤버로 가져와짐!!!!
        
        object_categories = parse_object_goal_instruction(object_nav_task.instruction) #* 여기서 LLM 적용된다, object_nav_task.instaruction은 for문 스텝에 따른 task_id에서의 이동 명령 자연어 ex. (Go to the closest cushion first, then go to a chair nearby, after that, go to a counter and in the end, navigate to a table. )
        print(f"instruction: {object_nav_task.instruction}")
        robot.empty_recorded_actions() #* LangRobot class에 있으며, 이동 명령 획득을 위한 초기 세팅 담당
        robot.set_agent_state(object_nav_task.init_hab_tf) #* HabitatLanguageRobot class에 있으며, 로봇의 위치를 설정해주는 함수이며, 여기선 초기 위치를 넣었기에 초기위치 세팅
                                                            #* init_hab_tf는 task_dict의 첫번째 task의 tf_habitat이며, 이는 habitat simulator의 변환행렬을 의미하는 듯
        for cat_i, cat in enumerate(object_categories): #* object_categories는 목표 키워드의 리스트이므로 cat_i에는 목표 키워드의 인덱스 번호, cat에는 해당 스텝에서의 목표 키워드가 들어감
            print(f"Navigating to category {cat}")
            actions_list = robot.move_to_object(cat) #* 이게 target 키워드로 이동할 때의 이동 명령의 리스트를 반환해주는 함수인듯
        recorded_actions_list = robot.get_recorded_actions()
        robot.set_agent_state(object_nav_task.init_hab_tf)
        for action in recorded_actions_list:
            object_nav_task.test_step(robot.sim, action, vis=config.nav.vis)

        save_dir = robot.vlmaps_dataloader.data_dir / (config.map_config.map_type + "_obj_nav_results")
        os.makedirs(save_dir, exist_ok=True)
        save_path = save_dir / f"{task_id:02}.json"
        object_nav_task.save_single_task_metric(save_path)
