from operator import truediv
from matplotlib.pyplot import close
from pydantic import BaseModel, Field
from ROAR.control_module.controller import Controller
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
# import keyboard

from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
from collections import deque
from enum import Enum
import numpy as np
import math
import logging
from ROAR.agent_module.agent import Agent
from typing import Tuple
import json
from pathlib import Path
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner

class SpeedData:
    def __init__(self, distance_to_section, current_speed, target_speed, recommended_speed):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed

class SlowDownPoint:
    def __init__(self, transform, targetSpeed, distance):
        self.transform = transform
        self.targetSpeed = targetSpeed
        self.distance = distance

class PIDFastController(Controller):
    # save debug messages to show after crash or finish.
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self, agent, steering_boundary: Tuple[float, float],
                 throttle_boundary: Tuple[float, float], **kwargs):
        super().__init__(agent, **kwargs)
        self.max_radius = 10000
        self.max_speed = self.agent.agent_settings.max_speed
        throttle_boundary = throttle_boundary
        self.steering_boundary = steering_boundary
        self.config = json.load(Path(agent.agent_settings.pid_config_file_path).open(mode='r'))
        self.intended_target_distance = [0, 30, 60, 90, 120, 150, 180] 
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0
        self.slowList = self.init_slow_points()

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False
        
        self.lat_pid_controller = LatPIDController(
            agent=agent,
            config=self.config["latitudinal_controller"],
            dt=0.05,
            steering_boundary=steering_boundary
        )
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        for s in self.__class__.debug_strings:
            print(s)

    def run_in_series(self, 
                      next_waypoint: Transform, 
                      close_waypoint: Transform,  # ok to remove 
                      far_waypoint: Transform,    # ok to remove
                      more_waypoints: [Transform], **kwargs) -> VehicleControl:

        # run lat pid controller
        # NOTE: ok to remove wide_error and sharp_error
        steering, error, wide_error, sharp_error = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint, close_waypoint=close_waypoint, far_waypoint=far_waypoint)

        self.tick_counter += 1
        throttle, brake = self.get_throttle_and_brake(more_waypoints)
        #throttle, brake = self._brake_test(throttle, brake)

        current_speed = Vehicle.get_speed(self.agent.vehicle)
        # calculate change in pitch
        pitch = float(next_waypoint.record().split(",")[4])
        gear = max(1, (int)((current_speed - 2*pitch) / 60))
        if throttle == -1:
            gear = -1

        # if keyboard.is_pressed("space"):
        #      print(self.agent.vehicle.transform.record())

        self.dprint("--- " + str(throttle) + " " + str(brake) 
                    + " steer " + str(steering)
                    + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
                    + " " + str(self.agent.vehicle.transform.location.z)) 

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        return VehicleControl(throttle=throttle, steering=steering, brake=brake, gear=gear)

    def init_slow_points(self):
        slowList = []
        p1 = self.string_to_transform("3017.900146484375,152.14938354492188,3765.33740234375,0.13231982290744781,3.9114372730255127,-64.34010887145996")
        slowList.append(SlowDownPoint(p1, 130, 20))

        p2 = self.string_to_transform("3151.341797,159.2339935,3716.354492,-0.321014404,5.717428207,-78.48518085")
        slowList.append(SlowDownPoint(p2, 130, 10))

        p3 = self.string_to_transform("4203.197754,491.6361694,2752.22998,-0.246795535,6.657794952,-106.3645782")
        slowList.append(SlowDownPoint(p3, 100, 15))

        p4 = self.string_to_transform("4432.474609,498.8491821,2793.180176,0.044999924,3.816627502,-116.8201389")
        slowList.append(SlowDownPoint(p4, 90, 15))

        p5 = self.string_to_transform("5613.11377,400.4781494,4202.975586,-0.189971924,-5.463495255,86.37561035")
        slowList.append(SlowDownPoint(p5, 43, 15))

        p6 = self.string_to_transform("5012.12646484375,322.0113525390625,3822.21142578125,-0.6661374568939209,7.25579833984375,34.138038635253906")
        slowList.append(SlowDownPoint(p6, 27, 15))

        p7 = self.string_to_transform("4915.248047,301.824585,3980.433105,-0.070221022,-8.900111198,144.3268738")
        slowList.append(SlowDownPoint(p7, 135, 8))

        return slowList
    
    def string_to_transform(self, transform_string: str):
        raw = transform_string.split(",")
        return Transform(location=Location(x=raw[0], y=raw[1], z=raw[2]), rotation=Rotation(pitch=0, yaw=0, roll=0))

    def get_throttle_and_brake(self, more_waypoints: [Transform]):
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        wp = self.get_next_interesting_waypoints(more_waypoints)
        r1 = self.get_radius(wp[self.close_index : self.close_index + 3])
        r2 = self.get_radius(wp[self.mid_index : self.mid_index + 3])
        r3 = self.get_radius(wp[self.far_index : self.far_index + 3])

        p1 = \
            math.asin( np.clip((wp[1].location.y - wp[0].location.y) / (self.target_distance[1] - self.target_distance[0]), -0.5, 0.5))
        p2 = \
            math.asin( np.clip((wp[2].location.y - wp[0].location.y) / (self.target_distance[2] - self.target_distance[0]), -0.5, 0.5))
        # taking minimum, to avoid underestimating pitch on downhill.
        pitch_ahead = min(p1, p2)

        target_speed1 = self.get_target_speed(r1, pitch_ahead)
        target_speed2 = self.get_target_speed(r2, pitch_ahead)
        target_speed3 = self.get_target_speed(r3, pitch_ahead)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]
        speed_data = []
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, pitch_ahead))
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, pitch_ahead))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, pitch_ahead))

        if current_speed > 220:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            r4 = self.get_radius([wp[self.close_index], wp[self.close_index+2], wp[self.close_index+4]])
            target_speed4 = self.get_target_speed(r4, pitch_ahead)
            speed_data.append(self.speed_for_turn(close_distance, target_speed4, pitch_ahead))

        slow_down = self.speed_for_slow_down(pitch_ahead)
        if slow_down is not None:
            speed_data.append(slow_down)

        update = self.select_speed(speed_data)

        t, b = self.speed_data_to_throttle_and_brake(update, pitch_ahead)
        return t, b

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData, pitch_ahead: float):
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now

        self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2)) 
                    + " ts= " + str(round(speed_data.target_speed_at_distance, 2)) 
                    + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))

        percent_change_per_tick = 0.07 # speed drop for one time-tick of braking
        speed_up_threshold = 0.99
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        debug_note = ""
        if pitch_ahead < math.radians(-6):
            percent_change_per_tick = 0.05
            speed_up_threshold = 0.94
            throttle_decrease_multiple = 0.4
            throttle_increase_multiple = 1.05
            debug_note += "-6"
        if debug_note != "":
            self.dprint("changing multiples- " + str(debug_note) + " " + str(pitch_ahead) + " deg " + str(math.degrees(pitch_ahead)) 
                    + " loc " + str(self.agent.vehicle.transform.location.x))

        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero

        if percent_of_max > 1:
            # Consider slowing down
            brake_threshold_multiplier = 3.0
            if speed_data.current_speed > 200:
                brake_threshold_multiplier = 2.0
            if percent_of_max > 1 + (brake_threshold_multiplier * percent_change_per_tick):
                if self.brake_ticks > 0:
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: counter" + str(self.brake_ticks))
                    return -1, 1
                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and not self.speed_dropping_fast(percent_change_per_tick):
                    # start braking, and set for how many ticks to brake
                    self.brake_ticks = math.floor((percent_of_max - 1) / percent_change_per_tick)
                    # TODO: try 
                    # self.brake_ticks = 1, or (1 or 2 but not more)
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: initiate counter" + str(self.brake_ticks))
                    return -1, 1
                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle early1: sp_ch=" + str(percent_speed_change))
                    self.brake_ticks = 0 # done slowing down. clear brake_ticks
                    return 1, 0
            else:
                if self.speed_dropping_fast(percent_change_per_tick):
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle early2: sp_ch=" + str(percent_speed_change))
                    self.brake_ticks = 0 # done slowing down. clear brake_ticks
                    return 1, 0
                throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed, pitch_ahead)
                if percent_of_max > 1.02 or percent_speed_change > (-percent_change_per_tick / 2):
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle down: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain * throttle_decrease_multiple, 0 # coast, to slow down
                else:
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle maintain: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0 # done slowing down. clear brake_ticks
            # Consider speeding up
            if self.speed_dropping_fast(percent_change_per_tick):
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle: full speed drop: sp_ch=" + str(percent_speed_change))
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle full: p_max=" + str(percent_of_max))
                return 1, 0
            throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed, pitch_ahead)
            if percent_of_max < 0.98 or percent_speed_change < -0.01:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle up: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain * throttle_increase_multiple, 0 
            else:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle maintain: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain, 0

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def speed_dropping_fast(self, percent_change_per_tick: float):
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        percent_speed_change = (current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    # find speed_data with smallest recommended speed (same as the largest speed excess [current > recommended])
    # TODO: change to look for smallest recommended speed.
    def select_speed(self, speed_data: [SpeedData]):
        largest_diff = -300
        index_of_largest_diff = -1
        for i, sd in enumerate(speed_data):
            if sd.speed_diff > largest_diff:
                largest_diff = sd.speed_diff
                index_of_largest_diff = i

        if index_of_largest_diff != -1:
            return speed_data[index_of_largest_diff]
        else:
            return speed_data[0]
    
    def get_throttle_to_maintain_speed(self, current_speed: float, pitch_ahead: float):
        # TODO: commpute throttle needed to maintain current speed with given pitch.
        #       need to consider current_speed
        throttle = 0.6 + current_speed/1000
        if pitch_ahead < math.radians(-4):
            throttle *= 0.95
        if pitch_ahead < math.radians(-6):
            throttle *= 0.95
        if pitch_ahead < math.radians(-8):
            throttle *= 0.93
        if pitch_ahead < math.radians(-10):
            throttle *= 0.92
        if pitch_ahead < math.radians(-11):
            throttle *= 0.92

        if pitch_ahead > math.radians(2):
            throttle *= 1.03
        if pitch_ahead < math.radians(3):
            throttle *= 1.03
        if pitch_ahead < math.radians(4):
            throttle *= 1.03
        if pitch_ahead < math.radians(5):
            throttle *= 1.03
        return throttle

    def speed_for_slow_down(self, pitch_ahead: float):
        for slowPoint in self.slowList:
            distance_to_speed_point = self.agent.vehicle.transform.location.distance(slowPoint.transform.location)
            if distance_to_speed_point < slowPoint.distance:
                self.dprint("\nspecial slow down point: ")
                self.dprint(slowPoint.transform)
                return self.speed_for_turn(2 + distance_to_speed_point/5, slowPoint.targetSpeed, pitch_ahead)

        return None

    def speed_for_turn(self, distance: float, target_speed: float, pitch_ahead: float):
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        d = (1/675) * (target_speed**2) + distance
        max_speed = math.sqrt(675 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def get_next_interesting_waypoints(self, more_waypoints: [Transform]):
        # return a list of points with distances approximately as given 
        # in intended_target_distance[] from the current location.
        points = []
        dist = [] # for debugging
        start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            curr_dist += start.location.distance(end.location)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " +  str(dist))
        return points

    def get_radius(self, wp: [Transform]):
        point1 = (wp[0].location.x, wp[0].location.z)
        point2 = (wp[1].location.x, wp[1].location.z)
        point3 = (wp[2].location.x, wp[2].location.z)

        # Calculating length of all three sides
        len_side_1 = round( math.dist(point1, point2), 3)
        len_side_2 = round( math.dist(point2, point3), 3)
        len_side_3 = round( math.dist(point1, point3), 3)
        small_num = 0.01
        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.max_radius

        # sp is semi-perimeter
        sp = (len_side_1 + len_side_2 + len_side_3) / 2

        # Calculating area using Herons formula
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        if area_squared < small_num:
            return self.max_radius
        # Calculating curvature using Menger curvature formula
        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))
        return radius

    def get_target_speed(self, radius: float, pitch=0.0):
        if radius >= self.max_radius:
            return self.max_speed
        mu = 1.16
        target_speed = math.sqrt(mu*9.81*radius) * 3.6
        return max(20, min(target_speed, self.max_speed))  # clamp between 20 and max_speed

    def print_speed(self, text: str, s1: float, s2: float, s3: float, curr_s: float):
        self.dprint(text + " s1= " + str(round(s1, 2)) + " s2= " + str(round(s2, 2)) + " s3= " + str(round(s3, 2))
            + " cspeed= " + str(round(curr_s, 2)))

    # debug print
    def dprint(self, text):
        if PIDFastController.display_debug:
            PIDFastController.debug_strings.append(text)

    # static debug print, to store debug text from LatPIDController
    @staticmethod
    def sdprint(text):
        if PIDFastController.display_debug:
            PIDFastController.debug_strings.append(text)

    # brake test. print time, speed, distance after hitting brakes at some initial speed. 
    # def run_in_series_brake_test(self, 
    #                   next_waypoint: Transform, 
    #                   close_waypoint: Transform, 
    #                   far_waypoint: Transform, 
    #                   more_waypoints: [Transform], **kwargs) -> VehicleControl:

    #     # run lat pid controller
    #     steering, error, wide_error, sharp_error = self.lat_pid_controller.run_in_series(next_waypoint=next_waypoint, close_waypoint=close_waypoint, far_waypoint=far_waypoint)
        
    #     current_speed = Vehicle.get_speed(self.agent.vehicle)
    #     throttle = 1
    #     brake = 0
    #     if current_speed > 250 and self.brake_test_counter == 0:
    #         throttle = -1
    #         brake = 1
    #         self.brake_test_counter = 1
    #         self.brake_start = self.agent.vehicle.transform
    #     elif self.brake_test_counter > 0:
    #         throttle = -1
    #         brake = 1
    #         self.brake_test_counter += 1
        
    #     if current_speed > 1 and self.brake_test_counter > 0 and (self.brake_test_counter) % 5 == 1:
    #         break_dist = self.brake_start.location.distance(self.agent.vehicle.transform.location)
    #         print("Break test: " + str(self.brake_test_counter - 1) + " s= " + str(round(current_speed, 1)) + " d= " + str(round(break_dist, 1)))
    #     gear = 1
    #     return VehicleControl(throttle=throttle, steering=steering, brake=brake, gear=gear)


    @staticmethod
    def find_k_values(vehicle: Vehicle, config: dict) -> np.array:
        current_speed = Vehicle.get_speed(vehicle=vehicle)
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

class LatPIDController(Controller):
    def __init__(self, agent, config: dict, steering_boundary: Tuple[float, float],
                 dt: float = 0.03, **kwargs):
        super().__init__(agent, **kwargs)
        self.config = config
        self.steering_boundary = steering_boundary
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, next_waypoint: Transform, close_waypoint: Transform, far_waypoint: Transform, **kwargs) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = self.agent.vehicle.transform.location.to_array()
        direction_vector = np.array([-np.sin(np.deg2rad(self.agent.vehicle.transform.rotation.yaw)),
                                     0,
                                     -np.cos(np.deg2rad(self.agent.vehicle.transform.rotation.yaw))])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), 0, (v_end[2] - v_begin[2])])
        
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location.x - v_begin[0],
                0,
                next_waypoint.location.z - v_begin[2],
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        #error = np.arccos(v_vec_normed @ w_vec_normed.T)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1)) # makes sure arccos input is between -1 and 1, inclusive
        _cross = np.cross(v_vec_normed, w_vec_normed)

        # calculate close error projection
        w_vec = np.array(
            [
                close_waypoint.location.x - v_begin[0],
                0,
                close_waypoint.location.z - v_begin[2],
            ]
        )
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        #wide_error = np.arccos(v_vec_normed @ w_vec_normed.T)
        wide_error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1)) # makes sure arccos input is between -1 and 1, inclusive

        # calculate far error projection
        w_vec = np.array(
            [
                far_waypoint.location.x - v_begin[0],
                0,
                far_waypoint.location.z - v_begin[2],
            ]
        )
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        #sharp_error = np.arccos(v_vec_normed @ w_vec_normed.T)
        sharp_error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1)) # makes sure arccos input is between -1 and 1, inclusive

        if _cross[1] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = PIDFastController.find_k_values(config=self.config, vehicle=self.agent.vehicle)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        
        PIDFastController.sdprint("steer: " + str(lat_control) + " err" + str(error) + " k_p=" + str(k_p) + " de" + str(_de) + " k_d=" + str(k_d) 
            + " ie" + str(_ie) + " k_i=" + str(k_i) + " sum" + str(sum(self._error_buffer)))

        return lat_control, error, wide_error, sharp_error



# PID config
# {
#     "longitudinal_controller": {
#       "40": {
#             "Kp": 0.8,
#             "Kd": 0.4,
#             "Ki": 0
#       },
#       "60": {
#             "Kp": 0.5,
#             "Kd": 0.3,
#             "Ki": 0
#       },
#       "90": {
#             "Kp": 0.3,
#             "Kd": 0.3,
#             "Ki": 0
#       },
#       "150": {
#             "Kp": 0.2,
#             "Kd": 0.2,
#             "Ki": 0.1
#           }
#     },
#     "latitudinal_controller": {
#       "60": {
#             "Kp": 0.8,
#             "Kd": 0.05,
#             "Ki": 0.05
#       },
#       "70": {
#             "Kp": 0.7,
#             "Kd": 0.07,
#             "Ki": 0.07
#       },
#       "80": {
#             "Kp": 0.66,
#             "Kd": 0.08,
#             "Ki": 0.08
#       },
#       "90": {
#             "Kp": 0.63,
#             "Kd": 0.09,
#             "Ki": 0.09
#       },
#       "100": {
#             "Kp": 0.6,
#             "Kd": 0.1,
#             "Ki": 0.1
#       },
#       "120": {
#             "Kp": 0.52,
#             "Kd": 0.1,
#             "Ki": 0.1
#       },
#       "130": {
#             "Kp": 0.51,
#             "Kd": 0.1,
#             "Ki": 0.09
#       },
#       "140": {
#             "Kp": 0.52,
#             "Kd": 0.1,
#             "Ki": 0.09
#       },
#       "160": {
#             "Kp": 0.5,
#             "Kd": 0.08,
#             "Ki": 0.06
#       },
#       "180": {
#             "Kp": 0.28,
#             "Kd": 0.02,
#             "Ki": 0.05
#       },
#       "200": {
#             "Kp": 0.28,
#             "Kd": 0.03,
#             "Ki": 0.04
#       },
#       "230": {
#             "Kp": 0.3,
#             "Kd": 0.04,
#             "Ki": 0.05
#       },
#       "300": {
#             "Kp": 0.205,
#             "Kd": 0.008,
#             "Ki": 0.017
#       }
#     }
#   }
