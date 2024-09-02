import os
import json


def get_object_text_info(traj_data):
    scene_num = traj_data['scene']['scene_num']
    object_list = []
    for i in traj_data['scene']['object_poses']:
        object_list.append(i['objectName'].split("_")[0].lower())
    openable_json_file = os.path.join(f'{your_path}' % scene_num)
    with open(openable_json_file, 'r') as f:
        openable_points = json.load(f)
    object_list = list(set(object_list)) + list(set([i.split("|")[0].lower() for i in list(openable_points.keys())]))
    if len(traj_data['scene']['object_toggles']) > 0:
        light = traj_data['scene']['object_toggles'][0]['objectType'].lower()
        object_list.append(light)
    obj_text = 'Objects present in the environment: ['
    for i in object_list:
        obj_text += i + ", "
    obj_text = obj_text[:-2] + "]. "
    return obj_text,object_list


def get_object_visual_info(split, folder, subfolder, traj_data):
    scene_num = traj_data['scene']['scene_num']
    file_path = os.path.join(f'{your_path}', split, folder, subfolder, 'FloorPlan%s-objects.json' % scene_num)
    with open(file_path, 'r') as f:
        blip_data = json.load(f)
    visual_text = 'Here are some description of the surronding of the objects, some might be hidden: ['
    for i in blip_data:
        if i['caption'] == '': continue
        if len(visual_text) > 200: break
        obj_name = i['object_id'].split("|")[0].lower()
        obj_surrounding = i['caption']
        visual_text += obj_name + ": " + obj_surrounding + ", "
    visual_text = visual_text[:-2] + "]. "
    return visual_text
