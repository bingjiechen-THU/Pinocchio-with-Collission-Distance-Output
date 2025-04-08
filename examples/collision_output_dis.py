"""
Collision detection and nearest distance between shapes are added.
"""

import coal
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import numpy as np
import time

from pyroboplan.core.utils import extract_cartesian_poses, set_collisions
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames


def prepare_collision_scene(model, collision_model):
    """Helper function to create a collision scene for this example."""

    # Modify the collision model so all the Panda links are translucent
    for cobj in collision_model.geometryObjects:
        cobj.meshColor = np.array([0.7, 0.7, 0.7, 0.25])

    # Table surface
    obstacle_1 = pinocchio.GeometryObject(
        "table",
        0,
        pinocchio.SE3(np.eye(3), np.array([0, 0.6, 0.3])),
        coal.Box(1, 0.6, 0.04),
    )
    obstacle_1.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_1)
    
    # The first table leg
    obstacle_2 = pinocchio.GeometryObject(
        "table_leg1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.48, 0.32, 0.15])),
        coal.Cylinder(0.02, 0.3),
    )
    obstacle_2.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_2)

    # The second table leg
    obstacle_3 = pinocchio.GeometryObject(
        "table_leg2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.48, 0.88, 0.15])),
        coal.Cylinder(0.02, 0.3),
    )
    obstacle_3.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_3)

    # The third table leg
    obstacle_4 = pinocchio.GeometryObject(
        "table_leg3",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.48, 0.32, 0.15])),
        coal.Cylinder(0.02, 0.3),
    )
    obstacle_4.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_4)

    # The fourth table leg
    obstacle_5 = pinocchio.GeometryObject(
        "table_leg4",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.48, 0.88, 0.15])),
        coal.Cylinder(0.02, 0.3),
    )
    obstacle_5.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_5)

    
    # Gets the ncollision name of the robotic arm
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    
    # The collision pair is activated so that each component and obstacle of the robot arm are combined into a collision pair one by one
    # only add the table as an obstacle (without the legs)
    obstacle_names = ["table"]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)



if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    
    # collision pairs of the robot arm self-collision
    # add_self_collisions(model, collision_model)
    
    prepare_collision_scene(model, collision_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayCollisions(True)
    viz.displayVisuals(False)

    # Define a joint space path
    q_start = pinocchio.randomConfiguration(model)
    q_end = pinocchio.randomConfiguration(model)
    max_step_size = 0.05
    q_path = discretize_joint_space_path([q_start, q_end], max_step_size)

    # Visualize the path
    target_tforms = extract_cartesian_poses(model, "panda_hand", q_path)
    visualize_frames(viz, "path", target_tforms, line_length=0.05, line_width=1)
    viz.display(q_start)
    time.sleep(0.5)
    input("Press 'Enter' to animate the path")

    # Collision check along the path
    for q in q_path:
        
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, False
        )
        
        dis = []
        contacts = []
        for k in range(len(collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = collision_model.collisionPairs[k]
            if cr.isCollision():
                print(
                    "collision between:",
                    collision_model.geometryObjects[cp.first].name,
                    " and ",
                    collision_model.geometryObjects[cp.second].name,
                )
                for contact in cr.getContacts():
                    contacts.extend(
                        [contact.getNearestPoint1(), contact.getNearestPoint2()]
                    )

            
            # Calculates the distance between collision pairs at the specified index k
            pinocchio.computeDistance(collision_model, collision_data, k)
            dis.append(collision_data.distanceResults[k].min_distance)
            print(collision_model.geometryObjects[cp.first].name, collision_model.geometryObjects[cp.second].name, dis[-1])
            
            # it seems that python has no way to directly retrieve the nearest point between collision pairs, because the returned cpp type is not converted in the python interface at pinocchi
            # print(f"Nearest point on object 1: {collision_data.distanceResults[k].nearest_points[0]}")
            # print(f"Nearest point on object 2: {collision_data.distanceResults[k].nearest_points[1]}")
   
            
        print("Obstacle Minimum Distance: ", min(dis))
            
        if len(contacts) == 0:
            print("Found no collisions!")


        viz.viewer["collision_display"].set_object(
            mg.LineSegments(
                mg.PointsGeometry(
                    position=np.array(contacts).T,
                    color=np.array([[1.0, 0.0, 0.0] for _ in contacts]).T,
                ),
                mg.LineBasicMaterial(
                    linewidth=3,
                    vertexColors=True,
                ),
            )
        )

        viz.display(q)
        time.sleep(0.1)
