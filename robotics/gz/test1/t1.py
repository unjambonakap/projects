import chdrft.utils.misc as cmisc
import pybullet as p
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.setGravity(0,0,-10)
 #p.setGravity(gravity[0], gravity[1], gravity[2])
    #p.setPhysicsEngineParameter(fixedTimeStep=1.0/60., numSolverIterations=5, numSubSteps=2)
planeId = p.loadSDF(cmisc.path_here("/home/benoit/repos/bullet3/data/two_cubes.sdf"))
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
p.stepSimulation()
#cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#print(cubePos,cubeOrn)
#p.disconnect()
import time
print('start')
for i in range(1000):
    time.sleep(0.1)
    print(i)
    p.stepSimulation()

