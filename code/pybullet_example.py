import numpy as np
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf", [0,0,0],[0,0,0,1])
targid = p.loadURDF("husky/husky.urdf",[0,0,0],[0,0,0,1])
obj_of_focus = targid

print("There are {} joint(s)".format(p.getNumJoints(targid)))
jointid = 4
jType = p.getJointInfo(targid, jointid)
jLower = p.getJointInfo(targid, jointid)[8]
jUpper = p.getJointInfo(targid, jointid)[9]


for step in range(500):
    joint_two_targ = np.random.uniform(jLower, jUpper)
    joint_four_targ = np.random.uniform(jLower, jUpper)
    p.setJointMotorControlArray(targid, [2,4], p.POSITION_CONTROL, targetPositions = [joint_two_targ, joint_four_targ])

    focus_position, _ = p.getBasePositionAndOrientation(targid)
    p.resetDebugVisualizerCamera(cameraDistance = 3, cameraYaw=0, cameraPitch=-40, cameraTargetPosition = focus_position)
    p.stepSimulation()
    time.sleep(.01)