import os
import shutil
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(script_dir, "result_files")

def file_processing_json():
    files = os.listdir(directory)
    files.sort()

    filenames = []
    file_contents = {}

    dim_list = []
    dimensions = {}

    for file in files:
        if file.endswith(".json"):
            if file.startswith("defect_pose"):
                filenames.append(file)
            else:
                dim_list.append(file)
                

    for filename in filenames:
        with open(directory + "\{}".format(filename), "r") as file:
            data = json.load(file)
            file_contents[filename] = data
    
    for dim in dim_list:
        with open(directory + "\{}".format(dim), "r") as file:
            box_dim = json.load(file)
            dimensions[dim] = box_dim

    return filenames, file_contents, dim_list, dimensions

def file_processing_pcd():
    files = os.listdir(directory)
    files.sort()
    defect_filenames = []
    defect_file_contents = {}
    object_file_content = {}

    for file in files:
        if file.endswith(".pcd"):
            if file.startswith("defect_pc"):
                defect_filenames.append(file)
            else: 
                with open(directory + "\{}".format(file), "r") as object_file:
                    object_file_content[file] = object_file

    for defect_name in defect_filenames:
        with open(directory + "\{}".format(defect_name), "r") as defect_file:
            defect_file_contents[defect_name] = defect_file

    return defect_filenames, defect_file_contents, object_file_content


def delete_directory_exclude_file(directory, excluded_file=""):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            if file_path != excluded_file:
                os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)