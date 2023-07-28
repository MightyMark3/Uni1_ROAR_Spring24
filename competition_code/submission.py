"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface



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

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.
        pass