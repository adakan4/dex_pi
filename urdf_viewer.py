import pybullet as p
import pybullet_data
import time

# Connect to the GUI
p.connect(p.GUI)

# Set search path to find URDFs and textures
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Optionally set gravity
p.setGravity(0, 0, -9.81)

# Load a plane so the robot has ground
plane_id = p.loadURDF("plane.urdf")

# Load your robot URDF
robot_id = p.loadURDF(
    "xarm/xarm6.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE
)

# Run simulation (keep GUI open)
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)