import numpy as np
import os
import json

from visualisation import *

script_dir = os.path.dirname(os.path.abspath(__file__)) 
directory =  os.path.join(script_dir, "result_files")
files = os.listdir(directory)
file_content = {}

raster_waypoints = plot()

'--------------------------DO NOT CONVERT DEFECT POSES TO QUATERNIONS IN MECH VISION, USE EULER--------------------------------'
for filename in files:
    if filename.startswith("defect_pose") & filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            file_content[filename] = data

script = """
    MODULE MainModule
    VAR robtarget home:= [[150.1, 210.4, 556.8],[0.0, 0.35559,-0.93464, 0.0],[-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    VAR robtarget p0:= [[400, 180, 400], [0.0, 0.35559,-0.93464, 0.0], [-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]]; 
"""

open("MAINMODULE.mod", "w").close()

with open(os.path.join(script_dir, "MAINMODULE.mod"), "a") as f:
    f.write(script)
    z_offset = 15 # in mm
    for i, (filename, data) in enumerate(file_content.items()):
        defect_pos = [x * 1000 if index < 2 else x * 1000 + 20 for index, x in enumerate(data[:3])]
        f.write("    VAR robtarget p{}:= [{}, [0.0, 0.35559,-0.93464, 0.0], [-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]]; \n".format(i + 1, defect_pos))

        defect_raster = raster_waypoints[i]
        for j in range(len(defect_raster)):
            f.write("    VAR robtarget p{}:= [[{}, {}, {}], [0.0, 0.35559,-0.93464, 0.0], [-1,-1,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]]; \n"
                    .format(100 * (i+1) + j, defect_raster[j][0] * 1000, defect_raster[j][1] * 1000, defect_raster[j][2] * 1000 + z_offset))

    
    
    f.write("    PERS tooldata tool1:=  [TRUE, [ [0, 0, 100], [1, 0, 0 ,0] ], [0.001, [0, 0, 0.001], [1, 0, 0, 0], 0, 0, 0]]; \n")
    f.write("    PERS wobjdata DefaultWObj; \n\n")
    f.write("    PROC main() \n        MoveL home, v50, z50, tool1; \n")


    for i in range(len(file_content)):
        f.write("        MoveL p{}, v50, z50, tool1; \n        WaitTime 3; \n".format(i+1))
        for j in range(len(raster_waypoints[i])):
            f.write("        MoveL p{}, v50, z50, tool1; \n".format(100* (i+1) + j))
        f.write("        WaitTime 3; \n")

    f.write("    ENDPROC \n")
    f.write("ENDMODULE")