"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
from manual_control import ManualControlViewer

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    async def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.
        self.manual_viewer = ManualControlViewer()

        
    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.
        vehicle_location = self.vehicle.get_3d_location()
        vehicle_rotation = self.vehicle.get_roll_pitch_yaw()

        # Receive camera data and render it
        camera_data = await self.camera.receive_observation()
        render_ret = self.manual_viewer.render(camera_data)
         # If user clicked the close button, render_ret will be None
        if render_ret is None:
            return
        
        way_points = self.maneuverable_waypoints
        # Find the waypoint closest to the vehicle
        current_waypoint_idx = filter_waypoints(
            vehicle_location,
            current_waypoint_idx,
            way_points
        )
         # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        waypoint_to_follow = way_points[(current_waypoint_idx + 3) % len(way_points)]

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = (
            -8.0 / np.sqrt(np.linalg.norm(self.vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
        ) if np.linalg.norm(self.vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
        steer_control = np.clip(steer_control, -1.0, 1.0)

        # Proportional controller to control the vehicle's speed towards 40 m/s
        throttle_control = 0.05 * (20 - np.linalg.norm(self.vehicle.get_linear_3d_velocity()))

        control = {
            "throttle": np.clip(throttle_control, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(-throttle_control, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
