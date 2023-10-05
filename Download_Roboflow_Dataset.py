from roboflow import Roboflow
rf = Roboflow(api_key="7dUHXSeVZVEs4T683Sow")
project = rf.workspace("moks-workspace").project("assault-detection-program")
dataset = project.version(4).download("yolov5")

