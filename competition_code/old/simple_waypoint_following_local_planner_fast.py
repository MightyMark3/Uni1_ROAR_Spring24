from ROAR.planning_module.local_planner.local_planner import LocalPlanner
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.control_module.controller import Controller
from ROAR.planning_module.mission_planner.mission_planner import MissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
# import keyboard

from functools import reduce
import itertools
import logging
from typing import Union
from ROAR.utilities_module.errors import (
    AgentException,
)
from ROAR.agent_module.agent import Agent
import json
from pathlib import Path
from statistics import mean


class SimpleWaypointFollowingLocalPlanner(LocalPlanner):
    def __init__(
            self,
            agent: Agent,
            controller: Controller,
            mission_planner: MissionPlanner,
            behavior_planner: BehaviorPlanner,
            closeness_threshold=0.5,
    ):
        """
        Initialize Simple Waypoint Following Planner
        Args:
            agent: newest agent state
            controller: Control module used
            mission_planner: mission planner used
            behavior_planner: behavior planner used
            closeness_threshold: how close can a waypoint be with the vehicle
        """
        super().__init__(agent=agent,
                         controller=controller,
                         mission_planner=mission_planner,
                         behavior_planner=behavior_planner,
                         )
        self.logger = logging.getLogger("SimplePathFollowingLocalPlanner")
        self.timing_section_index = [3000, 10000, 20000, 30000, 32000]
        self.timing_section_waypoints = []
        self.set_mission_plan()
        self.logger.debug("Simple Path Following Local Planner Initiated")
        self.closeness_threshold = closeness_threshold
        self.closeness_threshold_config = json.load(Path(
            agent.agent_settings.simple_waypoint_local_planner_config_file_path).open(mode='r'))
        self.previous_num_steps = 0

    def set_mission_plan(self) -> None:
        """
        Clears current waypoints, and reset mission plan from start
        I am simply transferring the mission plan into my waypoint queue.
        Assuming that this current run will run all the way to the end

        Returns:
            None
        """
        self.way_points_queue.clear()
        while (
                self.mission_planner.mission_plan
        ):  # this actually clears the mission plan!!
            self.way_points_queue.append(self.mission_planner.mission_plan.popleft())

        # self.way_points_queue = self.get_smoother_waypoints()
        
        if len(self.timing_section_waypoints) == 0: 
            for i in self.timing_section_index:
                if i < len(self.way_points_queue):
                    self.timing_section_waypoints.append(self.way_points_queue[i])

        # set waypoint queue to current spawn location
        # 1. find closest waypoint
        # 2. remove all waypoints prior to closest waypoint
        if False:
            closest_waypoint = self.way_points_queue[0]
            for waypoint in self.way_points_queue:
                cur_dist = self.agent.vehicle.transform.location.distance(waypoint.location)
                closest_dist = self.agent.vehicle.transform.location.distance(closest_waypoint.location)
                if  cur_dist < closest_dist:
                    closest_waypoint = waypoint
            while self.way_points_queue[0] != closest_waypoint:
                self.way_points_queue.popleft()

    def is_done(self) -> bool:
        """
        If there are nothing in self.way_points_queue,
        that means you have finished a lap, you are done

        Returns:
            True if Done, False otherwise
        """
        if len(self.way_points_queue) == 0:
            elapsed_time_steps = self.agent.time_counter - self.previous_num_steps
            print("\n\nFinished total: " + str(self.agent.time_counter) + " section: " + str(elapsed_time_steps))

        return len(self.way_points_queue) == 0

    def run_in_series(self) -> VehicleControl:
        """
        Run step for the local planner
        Procedure:
            1. Sync data
            2. get the correct look ahead for current speed
            3. get the correct next waypoint
            4. feed waypoint into controller
            5. return result from controller

        Returns:
            next control that the local think the agent should execute.
        """
        if (
                len(self.mission_planner.mission_plan) == 0
                and len(self.way_points_queue) == 0
        ):
            return VehicleControl()

        # get vehicle's location
        vehicle_transform: Union[Transform, None] = self.agent.vehicle.transform
        if vehicle_transform is None or type(vehicle_transform) != Transform:
            raise AgentException("I do not know where I am, I cannot proceed forward")

        # redefine closeness level based on speed
        self.set_closeness_threshold(self.closeness_threshold_config)

        # get current waypoint
        curr_closest_dist = float("inf")
        while True:
            if len(self.way_points_queue) == 0:
                self.logger.info("Destination reached")
                return VehicleControl()
            waypoint: Transform = self.way_points_queue[0]
            #print(waypoint)
            curr_dist = vehicle_transform.location.distance(waypoint.location)
            if curr_dist < curr_closest_dist:
                # if i find a waypoint that is closer to me than before
                # note that i will always enter here to start the calculation for curr_closest_dist
                curr_closest_dist = curr_dist
            elif curr_dist < self.closeness_threshold:
                # i have moved onto a waypoint, remove that waypoint from the queue
                passed_waypoint = self.way_points_queue.popleft()
                self.update_section_time(passed_waypoint)
            else:
                break
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        # target_waypoint = self.way_points_queue[0]
        target_waypoint = self.next_waypoint_smooth(current_speed)

        # if keyboard.is_pressed("t"):
        #     print(target_waypoint.record())
        #     print(self.agent.vehicle.transform.location)
        #     print(self.agent.vehicle.transform.record())
        #     pass
        # if keyboard.is_pressed("l"):
        #     print(vehicle_transform.location)

        waypoint_lookahead = round(pow(current_speed, 2)*0.002 + 0.7*current_speed)
        far_waypoint = self.way_points_queue[waypoint_lookahead]
        close_waypoint = self.way_points_queue[min(120, waypoint_lookahead)]
        more_waypoints = list(itertools.islice(self.way_points_queue, 0, 1000))
        # self.print_distances(target_waypoint, close_waypoint, far_waypoint)
        
        control: VehicleControl = self.controller.run_in_series(
            next_waypoint=target_waypoint, close_waypoint=close_waypoint, far_waypoint=far_waypoint, more_waypoints=more_waypoints)
        
        # self.logger.debug(f"\n"
        #                   f"Curr Transform: {self.agent.vehicle.transform}\n"
        #                   f"Target Location: {target_waypoint.location}\n"
        #                   f"Control: {control} | Speed: {Vehicle.get_speed(self.agent.vehicle)}\n")
        return control

    def set_closeness_threshold(self, config: dict):
        curr_speed = Vehicle.get_speed(self.agent.vehicle)
        for speed_upper_bound, closeness_threshold in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if curr_speed < speed_upper_bound:
                self.closeness_threshold = closeness_threshold
                break

                # Closeness threshold based on speed
                #     {
                #     "70": 12,
                #     "90": 14,
                #     "110": 16,
                #     "130": 19,
                #     "160": 22,
                #     "180": 32,
                #     "200": 42,
                #     "300": 52
                #   }

    def restart(self):
        self.set_mission_plan()

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py
    def next_waypoint_smooth(self, current_speed: float) -> (Transform):
        if current_speed > 70 and current_speed < 200 and self.agent.time_counter > 700:
            target_waypoint = self.average_point(self.closeness_threshold)
        else:
            target_waypoint = self.way_points_queue[0]
        return target_waypoint

    def average_point(self, num_points: int):
        smooth_lookahead = min(num_points, len(self.way_points_queue) - 1)

        sample_points = range(0, num_points)
        if smooth_lookahead > 100:  # Reduce computation by only looking at every 10 steps ahead
            sample_points = range(0, num_points, num_points // 10)
        if num_points > 5:
            location_sum = reduce(lambda x, y: x + y,
                                  (self.way_points_queue[i].location for i in sample_points))
            rotation_sum = reduce(lambda x, y: x + y,
                                  (self.way_points_queue[i].rotation for i in sample_points))

            num_points = len(sample_points)
            target_waypoint = Transform(location=location_sum / num_points, rotation=rotation_sum / num_points)
        else:
            target_waypoint = self.way_points_queue[0]

        return target_waypoint

    def update_section_time(self, passed_waypoint: Transform):
        if passed_waypoint in self.timing_section_waypoints:
            ind = self.timing_section_waypoints.index(passed_waypoint)
            elapsed_time_steps = self.agent.time_counter - self.previous_num_steps
            # print("\n\nFinished section " + str(ind) + " total: " + str(self.agent.time_counter) + " section: " + str(elapsed_time_steps))
            self.previous_num_steps = self.agent.time_counter

    def get_distance(self, wp: [Transform]):
        curr_dist = 0
        for i in range(len(wp) - 1):
            start = wp[i]
            end = wp[i+1]
            curr_dist += start.location.distance(end.location)
        return curr_dist
    
    def print_distances(self, next_wp: Transform, close_wp: Transform, far_wp: Transform):
        if self.agent.time_counter % 10 == 0:
            curr_wp = self.agent.vehicle.transform
            next_dist = curr_wp.location.distance(next_wp.location)
            close_dist = curr_wp.location.distance(close_wp.location)
            far_dist = curr_wp.location.distance(far_wp.location)
            print("Dist " + str(round(next_dist, 2)) + " " + str(round(close_dist)) + " " + str(round(far_dist)))

    def __del__(self):
        # in case of crash print where.
        if len(self.way_points_queue) > 0:
            print("\nNext waypoint is: " + str(self.way_points_queue[0]))