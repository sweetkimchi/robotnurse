import os
import os.path as osp
import time

import pybullet as p

# pip install tirmesh[easy]
import trimesh


p.connect(p.GUI)

root_dir = "YCB_Video_Models"
for i, model_dir in enumerate(os.listdir(root_dir)[:10]):
    model_dir = osp.join(root_dir, model_dir)
    obj_file = osp.join(model_dir, "textured_simple.obj")
    mesh = trimesh.load_mesh(obj_file)
    mesh.apply_translation([i * 0.2 - 1, 0, 0])
    points = mesh.vertices
    colors = mesh.visual.to_color().vertex_colors
    p.addUserDebugPoints(points[:5000], colors[:5000, :3] / 255)

while True:
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()
