import genesis as gs
import numpy as np


########################## Init ##########################
gs.init()

########################## Create a Scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=40,
        #gravity=(0.0, 0.0, 0.0)
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(1280, 720),
        max_FPS=60,
    ),
    show_viewer=True,
)

########################## Entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)

cube = scene.add_entity(
    material=gs.materials.Rigid(
        rho=80.0,
        friction=3.5,
    ),
    morph=gs.morphs.Box(
        size=(2.5, 1.0, 0.3),
        pos=(0.0, 0.0, 0.2), #-0.33
    ),
     surface=gs.surfaces.Default(
        color=(0.8, 0.4, 0.2, 1.0),
        vis_mode="collision",
    ),
)
#scene.show_contacts(True)  # or whatever toggle Genesis uses

# when loading an entity, you can specify its pose in the morph.
franka1 = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos   = (0.0, 0.7, 0.0),
        euler = (0, 0, 0),
    ),
)
franka2= scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos   = (0.0, -0.7, 0.0),#(0.6, -0.55, 0.0)
        euler = (0, 0, 0),
    ),
)
cloth= scene.add_entity(
    material=gs.materials.PBD.Cloth(),
    morph=gs.morphs.Mesh(
        file="meshes/cloth.obj",
        scale=1.5,
        pos=(0, 0, 0.8),
        euler=(0.0, 0, 0.0),
    ),
    surface=gs.surfaces.Default(
        color=(0.2, 0.4, 0.8, 1.0),
        vis_mode="visual",
    ),
)
scene.build()
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)
franka1.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka1.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka1.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)
franka2.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka2.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka2.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)
end_effector1 = franka1.get_link('hand')
end_effector2 = franka2.get_link('hand')
observed_x = np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0]])
# print(end_effector1.get_pos())

#cube.set_friction(0.2)


#plan to a point near the cloth1 
qpos = franka1.inverse_kinematics(
    link = end_effector1,
    pos  = np.array([0.4, 0.45, 0.5]),
    quat = np.array([0.0, 0.0, -0.9239, 0.3827]), #(0,1,0,0)(x,y,z,w)
)

qpos[-2:] = 0.01
path = franka1.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
qpos2 = franka2.inverse_kinematics(
    link = end_effector2,
    pos  = np.array([0.4, -0.45, 0.5]),
    quat = np.array([0.0, 0.0, 0.9239, 0.3827]), #(0,1,1,0)
)
qpos2[-2:] = 0.01
path2 = franka2.plan_path(
    qpos_goal     = qpos2,
    num_waypoints = 200, # 2s duration
)

# execute the planned path
for waypoint1,waypoint2 in zip(path,path2):
    franka1.control_dofs_position(waypoint1)
    franka2.control_dofs_position(waypoint2)
    scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()



#reach
qpos = franka1.inverse_kinematics(
    link = end_effector1,
    pos  = np.array([0.40, 0.45, 0.40]),
    quat=np.array([0.0, 0.0, -0.9239, 0.3827]),
)
qpos2 = franka2.inverse_kinematics(
    link = end_effector2,
    pos  = np.array([0.40, -0.45, 0.40]),
    quat = np.array([0.0, 0.0, 0.9239, 0.3827]),
)
franka1.control_dofs_position(qpos[:-2], motors_dof)
franka2.control_dofs_position(qpos2[:-2], motors_dof)
for i in range(50):
    scene.step()
# grasp
# franka1.control_dofs_position(qpos[:-2], motors_dof)
# franka1.control_dofs_force(np.array([-0.1, -0.1]), fingers_dof)

# for i in range(100):
#     scene.step()
# settling_steps = 100  # Number of steps to wait before starting MPPI
# print("Letting the cloth settle under gravity...")

# for i in range(settling_steps):
#     scene.step()  # Simulate one timestep with only gravity

import pandas as pd
csv_data = pd.read_csv('corner_velocities1.csv')
trajectory = []
trajectory1 = []
for index, row in csv_data.iterrows():
    vx = row['corner1_x']
    vy = row['corner1_y']
    vz = row['corner1_z']
    vx1 = row['corner2_x']
    vy1 = row['corner2_y']
    vz1 = row['corner2_z']
    velocity_vector = np.array([vx, vy, vz, 0, 0, 0])  
    velocity_vector1 = np.array([vx1, vy1, vz1, 0, 0, 0])
    trajectory.append(velocity_vector)
    trajectory1.append(velocity_vector1)
print(trajectory1)
for vel,vel1 in zip(trajectory,trajectory1):
    for step in range(80):
        particle_position = tuple(observed_x[0])
        particle_position1 = tuple(observed_x[1])
        particle = cloth.find_closest_particle(particle_position)
        particle1 = cloth.find_closest_particle(particle_position1)  
        J1 = franka1.get_jacobian(end_effector1)
        J2 = franka2.get_jacobian(end_effector2)
        Jinv = np.linalg.pinv(J1.cpu().numpy()) 
        Jinv1 = np.linalg.pinv(J2.cpu().numpy())
        dq = Jinv @ vel
        dq1 = Jinv1 @ vel1
        franka1.control_dofs_velocity(dq[:7], motors_dof)
        franka2.control_dofs_velocity(dq1[:7], motors_dof)
        cloth.set_particle_velocity(particle, vel[:3])
        cloth.set_particle_velocity(particle1, vel1[:3])
        pos = cloth.get_particles()  
        observed_x[0]=pos[particle]
        observed_x[1]=pos[particle1]
        scene.step()
    print("hi")


# # lift
# qpos = franka1.inverse_kinematics(
#     link=end_effector1,
#     pos=np.array([0.30, 0.30, 0.4]),
#     quat=np.array([0,1,0,0]),
# )
# franka1.control_dofs_position(qpos[:-2], motors_dof)
# for i in range(200):
#     scene.step()


while True:
    scene.step()