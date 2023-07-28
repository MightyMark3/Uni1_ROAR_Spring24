import roar_py_interface
import roar_py_carla
from submission import RoarCompetitionSolution
from infrastructure import RoarCompetitionAgentWrapper
from typing import List, Type, Optional, Dict, Any
import carla
import numpy as np
import gymnasium as gym
import asyncio

class RoarCompetitionRule:
    def __init__(
        self,
        waypoints : List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_carla.RoarPyCarlaActor,
        world: roar_py_carla.RoarPyCarlaWorld
    ) -> None:
        self.waypoints = waypoints
        self.waypoint_occupancy = np.zeros(len(waypoints),dtype=np.bool_)
        self.vehicle = vehicle
        self.world = world
        self._last_vehicle_location = vehicle.get_3d_location()
    
    def lap_finished(
        self
    ):
        return np.all(self.waypoint_occupancy)

    def tick(
        self
    ):
        current_location = self.vehicle.get_3d_location()
        delta_vector = current_location - self._last_vehicle_location
        delta_vector_norm = np.linalg.norm(delta_vector)
        delta_vector_unit = delta_vector / delta_vector_norm
        for i,waypoint in enumerate(self.waypoints):
            waypoint_delta = waypoint.location - current_location
            projection = np.dot(waypoint_delta,delta_vector_unit)
            projection = np.clip(projection,0,delta_vector_norm)
            closest_point_on_segment = current_location + projection * delta_vector_unit
            distance = np.linalg.norm(waypoint.location - closest_point_on_segment)
            if distance < 1.0:
                self.waypoint_occupancy[i] = True
        self._last_vehicle_location = current_location
    
    async def respawn(
        self
    ):
        vehicle_location = self.vehicle.get_3d_location()
    
        closest_waypoint_dist = np.inf
        closest_waypoint_idx = 0
        for i,waypoint in enumerate(self.waypoints):
            waypoint_dist = np.linalg.norm(vehicle_location - waypoint.location)
            if waypoint_dist < closest_waypoint_dist:
                closest_waypoint_dist = waypoint_dist
                closest_waypoint_idx = i
        closest_waypoint = self.waypoints[closest_waypoint_idx]
        closest_waypoint_location = closest_waypoint.location
        closest_waypoint_rpy = closest_waypoint.roll_pitch_yaw
        self.vehicle.set_transform(
            closest_waypoint_location + self.vehicle.bounding_box.extent[2] + 0.2, closest_waypoint_rpy
        )
        self.vehicle.set_linear_3d_velocity(np.zeros(3))
        self.vehicle.set_angular_velocity(np.zeros(3))
        for _ in range(20):
            await self.world.step()
        
        self._last_vehicle_location = self.vehicle.get_3d_location()

async def evaluate_solution(
    world : roar_py_carla.RoarPyCarlaWorld,
    solution_constructor : Type[RoarCompetitionSolution],
    max_seconds = 1200,
    # enable_visualization : bool = False TODO: implement visualization
) -> Optional[Dict[str, Any]]:
    waypoints = world.maneuverable_waypoints
    vehicle = world.spawn_vehicle(
        "vehicle.dallara.dallara",
        waypoints[0].location + 0.5,
        waypoints[0].roll_pitch_yaw,
        True,
    )
    assert vehicle is not None

    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]), # relative position
        np.array([0, 10/180.0*np.pi, 0]), # relative rotation
        image_width=1024,
        image_height=768
    )
    location_sensor = vehicle.attach_location_in_world_sensor()
    velocity_sensor = vehicle.attach_velocimeter_sensor()
    rpy_sensor = vehicle.attach_roll_pitch_yaw_sensor()
    occupancy_map_sensor = vehicle.attach_occupancy_map_sensor(
        50,
        50,
        2.0,
        2.0
    )
    collision_sensor = vehicle.attach_collision_sensor(
        np.zeros(3),
        np.zeros(3)
    )

    assert camera is not None
    assert location_sensor is not None
    assert velocity_sensor is not None
    assert rpy_sensor is not None
    assert occupancy_map_sensor is not None
    assert collision_sensor is not None

    solution : RoarCompetitionSolution = await solution_constructor(
        waypoints,
        RoarCompetitionAgentWrapper(vehicle),
        camera,
        location_sensor,
        velocity_sensor,
        rpy_sensor,
        occupancy_map_sensor,
        collision_sensor
    )
    rule = RoarCompetitionRule(waypoints,vehicle,world)

    for i in range(20):
        await world.step()
    
    start_time = world.last_tick_elapsed_seconds
    current_time = start_time
    
    while True:
        current_time = world.last_tick_elapsed_seconds
        if current_time - start_time > max_seconds:
            vehicle.close()
            return None
        await vehicle.receive_observation()

        if collision_sensor.get_last_observation().impulse_normal > 100.0:
            await rule.respawn()
        
        rule.tick()
        if rule.lap_finished():
            break
        
        await solution.step()
        await world.step()
    
    end_time = world.last_tick_elapsed_seconds
    vehicle.close()
    return {
        "elapsed_time" : end_time - start_time,
    }

async def main():
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(10.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    world = roar_py_instance.world
    world.set_control_steps(0.05, 0.005)
    world.set_asynchronous(False)
    evaluation_result = await evaluate_solution(
        world,
        RoarCompetitionSolution,
        max_seconds=1200
    )
    if evaluation_result is not None:
        print("Solution finished in {} seconds".format(evaluation_result["elapsed_time"]))
    else:
        print("Solution failed to finish in time")

if __name__ == "__main__":
    asyncio.run(main())