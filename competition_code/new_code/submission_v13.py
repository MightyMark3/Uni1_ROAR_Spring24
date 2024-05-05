"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface


def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def distance_p_to_p(p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

def new_x_y(x, y):
        new_location = np.array([x, y, 0])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=np.array([0,0,0]), 
                                                lane_width=5)

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        # self.maneuverable_waypoints = maneuverable_waypoints[:1962] + maneuverable_waypoints[1967:]
        # startInd = 1953
        startInd = 1800
        #endInd = 2006
        endInd = 1967
        # endInd = startInd+len(NEW_WAYPOINTS)
        self.maneuverable_waypoints = \
            maneuverable_waypoints[:startInd] + NEW_WAYPOINTS + maneuverable_waypoints[endInd:]
        # self.maneuverable_waypoints = self.modified_points(maneuverable_waypoints)
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_pid_controller = LatPIDController(config=self.get_lateral_pid_config())
        self.throttle_controller = ThrottleController()
        self.section_indeces = []
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.current_section = -1

    async def initialize(self) -> None:
        num_sections = 12
        #indexes_per_section = len(self.maneuverable_waypoints) // num_sections
        #self.section_indeces = [indexes_per_section * i for i in range(0, num_sections)]
        self.section_indeces = [198, 438, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"indexes: {self.section_indeces}")

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

    # def modified_points(self, waypoints):
    #     new_points = []
    #     for ind, waypoint in enumerate(waypoints):
    #         if ind == 1964:
    #             new_points.append(self.new_x(waypoint, -151))
    #         elif ind == 1965:
    #             new_points.append(self.new_x(waypoint, -153))
    #         elif ind == 1966:
    #             new_points.append(self.new_x(waypoint, -155))
    #         else:
    #             new_points.append(waypoint)
    #     return new_points
        
    def modified_points_bad(self, waypoints):
        end_ind = 1964
        num_points = 50
        start_ind = end_ind - num_points
        shift_vector = np.array([0.5, 0, 0])
        step_vector = shift_vector / num_points

        s2 = 1965
        num_points2 = 150
        shift_vector2 = np.array([0, 2.0, 0])


        new_points = []
        for ind, waypoint in enumerate(waypoints):
            p = waypoint
            if ind >= start_ind and ind < end_ind:
                p = self.point_plus_vec(p, step_vector * (ind - start_ind))
            if ind >= s2 and ind < s2 + num_points2:
                 p = self.point_plus_vec(p, shift_vector2)
            new_points.append(p)
        return new_points

    def modified_points_good(self, waypoints):
        start_ind = 1920
        num_points = 100
        end_ind = start_ind + num_points
        shift_vector = np.array([2.8, 0, 0])
        step_vector = shift_vector / num_points

        s2 = 1965
        num_points2 = 150
        shift_vector2 = np.array([0, 3.5, 0])

        s3 = 1920
        num_points3 = 195
        shift_vector3 = np.array([0.0, 0, 0])

        new_points = []
        for ind, waypoint in enumerate(waypoints):
            p = waypoint
            if ind >= start_ind and ind < end_ind:
                p = self.point_plus_vec(p, step_vector * (end_ind - ind))
                # p = self.point_plus_vec(p, step_vector * (end_ind - ind))
            if ind >= s2 and ind < s2 + num_points2:
                p = self.point_plus_vec(p, shift_vector2)
            if ind >= s3 and ind < s3 + num_points3:
                p = self.point_plus_vec(p, shift_vector3)
            new_points.append(p)
        return new_points

    def point_plus_vec(self, waypoint, vector):
        new_location = waypoint.location + vector
        # new_location = np.array([waypoint.location[0], new_y, waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location,
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw,
                                                lane_width=waypoint.lane_width)


    def modified_points_also_bad(self, waypoints):
        new_points = []
        for ind, waypoint in enumerate(waypoints):
            if ind >= 1962 and ind <= 2027:
                new_points.append(self.new_point(waypoint, self.new_y(waypoint.location[0])))
            else:
                new_points.append(waypoint)
        return new_points
    

    def new_x(self, waypoint, new_x):
        new_location = np.array([new_x, waypoint.location[1], waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw, 
                                                lane_width=waypoint.lane_width)
    def new_point(self, waypoint, new_y):
        new_location = np.array([waypoint.location[0], new_y, waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw, 
                                                lane_width=waypoint.lane_width)
    def new_y(self, x):

        y = -math.sqrt(102**2 - (x + 210)**2) - 962
        #print(str(x) + ',' + str(y))
        return y
        
        # a=0.000322627
        # b=2.73377
        # y = a * ( (abs(x + 206))**b ) - 1063.5
        # return y

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # compute and print section timing
        for i, section_ind in enumerate(self.section_indeces):
            if section_ind -2 <= self.current_waypoint_idx \
                and self.current_waypoint_idx <= section_ind + 2 \
                    and i != self.current_section:
                elapsed_ticks = self.num_ticks - self.section_start_ticks
                self.section_start_ticks = self.num_ticks
                self.current_section = i
                print(f"Section {i}: {elapsed_ticks}")

        new_waypoint_index = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh)
        #waypoint_to_follow = self.maneuverable_waypoints[new_waypoint_index]

        # Proportional controller to steer the vehicle
        steer_control = self.lat_pid_controller.run(
            vehicle_location, vehicle_rotation, current_speed_kmh, self.current_section, waypoint_to_follow)

        # Proportional controller to control the vehicle's speed
        waypoints_for_throttle = \
            (self.maneuverable_waypoints + self.maneuverable_waypoints)[new_waypoint_index:new_waypoint_index + 300]
        throttle, brake, gear = self.throttle_controller.run(
            self.current_waypoint_idx, waypoints_for_throttle, vehicle_location, current_speed_kmh, self.current_section)

        control = {
            "throttle": np.clip(throttle, 0.0, 1.0),
            "steer": np.clip(steer_control, -1.0, 1.0),
            "brake": np.clip(brake, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear
        }

        # print("--- " + str(throttle) + " " + str(brake) 
        #             + " steer " + str(steer_control)
        #             + " loc: " + str(vehicle_location)
        #             + " cur_ind: " + str(self.current_waypoint_idx)
        #             + " cur_sec: " + str(self.current_section)
        #             ) 


        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        speed_to_lookahead_dict = {
            70: 12,
            90: 12,
            110: 13,
            130: 14,
            160: 16,
            180: 20,
            200: 24,
            300: 24
        }
        num_waypoints = 3
        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
              num_waypoints = num_points
              break
        # if self.current_section in [8, 9]:
        #     # num_waypoints = num_waypoints // 2
        #     num_waypoints = 3 + (num_waypoints // 2)
        return num_waypoints

    def get_lookahead_index(self, speed):
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed) 
        #       + " cur_ind " + str(self.current_waypoint_idx) 
        #       + " num_points " + str(num_waypoints) 
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)
    
    def get_lateral_pid_config(self):
        conf = {
        "60": {
                "Kp": 0.8,
                "Kd": 0.05,
                "Ki": 0.05
        },
        "70": {
                "Kp": 0.7,
                "Kd": 0.07,
                "Ki": 0.07
        },
        "80": {
                "Kp": 0.66,
                "Kd": 0.08,
                "Ki": 0.08
        },
        "90": {
                "Kp": 0.63,
                "Kd": 0.09,
                "Ki": 0.09
        },
        "100": {
                "Kp": 0.6,
                "Kd": 0.1,
                "Ki": 0.1
        },
        "120": {
                "Kp": 0.52,
                "Kd": 0.1,
                "Ki": 0.1
        },
        "130": {
                "Kp": 0.51,
                "Kd": 0.1,
                "Ki": 0.09
        },
        "140": {
                "Kp": 0.45,
                "Kd": 0.1,
                "Ki": 0.09
        },
        "160": {
                "Kp": 0.4,
                "Kd": 0.05,
                "Ki": 0.06
        },
        "180": {
                "Kp": 0.28,
                "Kd": 0.02,
                "Ki": 0.05
        },
        "200": {
                "Kp": 0.28,
                "Kd": 0.03,
                "Ki": 0.04
        },
        "230": {
                "Kp": 0.26,
                "Kd": 0.04,
                "Ki": 0.05
        },
        "300": {
                "Kp": 0.205,
                "Kd": 0.008,
                "Ki": 0.017
        }
        }
        return conf

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py
    def next_waypoint_smooth(self, current_speed: float):
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]
        return target_waypoint

    def average_point(self, current_speed):
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2
        
        if self.current_section in []:
            num_points = lookahead_value
        if self.current_section in [8,9]:
            # num_points = lookahead_value // 2
            num_points = lookahead_value * 2
            # num_points = lookahead_value
            # num_points = 1
        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(self.maneuverable_waypoints)

        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location
  
        sample_points = [(start_index_for_avg + i) % len(self.maneuverable_waypoints) for i in range(0, num_points)]
        if num_points > 3:
            location_sum = reduce(lambda x, y: x + y,
                                  (self.maneuverable_waypoints[i].location for i in sample_points))
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section in [1,2]:
                max_shift_distance = 0.15
            if self.current_section in [8,9]:
                max_shift_distance = 2.0
            if self.current_section in [10,11,12]:
                max_shift_distance = 0.2
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv*max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                               roll_pitch_yaw=np.ndarray([0, 0, 0]), 
                                                               lane_width=0.0)
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location) 
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint =  self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint

class LatPIDController():
    def __init__(self, config: dict, dt: float = 0.05):
        self.config = config
        self.steering_boundary = (-1.0, 1.0)
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run(self, vehicle_location, vehicle_rotation, current_speed, cur_section, next_waypoint) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])
        
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location[0] - v_begin[0],
                next_waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1)) # makes sure arccos input is between -1 and 1, inclusive
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[2] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.find_k_values(cur_section, current_speed=current_speed, config=self.config)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        
        # PIDFastController.sdprint("steer: " + str(lat_control) + " err" + str(error) + " k_p=" + str(k_p) + " de" + str(_de) + " k_d=" + str(k_d) 
        #     + " ie" + str(_ie) + " k_i=" + str(k_i) + " sum" + str(sum(self._error_buffer)))
        # print("cross " + str(_cross))
        # print("loc " + str(vehicle_location) + " rot "  + str(vehicle_rotation))
        # print(" next.loc " + str(next_waypoint.location))

        # print("steer: " + str(lat_control) + " speed: " + str(current_speed) + " err" + str(error) + " k_p=" + str(k_p) + " de" + str(_de) + " k_d=" + str(k_d) 
        #     + " ie" + str(_ie) + " k_i=" + str(k_i) + " sum" + str(sum(self._error_buffer)))
        # print("   err P " + str(k_p * error) + " D " + str(k_d * _de) + " I " + str(k_i * _ie))

        return lat_control
    
    def find_k_values(self, cur_section, current_speed: float, config: dict) -> np.array:
        k_p, k_d, k_i = 1, 0, 0
        if cur_section in [8, 9, 10]:
        #   return np.array([0.3, 0.1, 0.25]) # ok for mu=1.2
        #   return np.array([0.2, 0.03, 0.15])
        #   return np.array([0.3, 0.06, 0.03]) # ok for mu=1.8
        #   return np.array([0.42, 0.05, 0.02]) # ok for mu=2.0
        #   return np.array([0.45, 0.05, 0.02]) # ok for mu=2.2
          return np.array([0.59, 0.05, 0.02]) # 

        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

    
    def normalize_rad(rad : float):
        return (rad + np.pi) % (2 * np.pi) - np.pi

class SpeedData:
    def __init__(self, distance_to_section, current_speed, target_speed, recommended_speed):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed

class ThrottleController():
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")
        # for s in self.__class__.debug_strings:
        #     print(s)

    def run(self, cur_wp_index, waypoints, current_location, current_speed, current_section) -> (float, float, int):
        self.tick_counter += 1
        throttle, brake = self.get_throttle_and_brake(cur_wp_index, current_location, current_speed, current_section, waypoints)
        gear = max(1, (int)(current_speed / 60))
        if throttle == -1:
            gear = -1

        # self.dprint("--- " + str(throttle) + " " + str(brake) 
        #             + " steer " + str(steering)
        #             + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
        #             + " " + str(self.agent.vehicle.transform.location.z)) 

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        # throttle = 0.05 * (100 - current_speed)
        return throttle, brake, gear

    def get_throttle_and_brake(self, cur_wp_index, current_location, current_speed, current_section, waypoints):

        wp = self.get_next_interesting_waypoints(current_location, waypoints)
        r1 = self.get_radius(wp[self.close_index : self.close_index + 3])
        r2 = self.get_radius(wp[self.mid_index : self.mid_index + 3])
        r3 = self.get_radius(wp[self.far_index : self.far_index + 3])

        target_speed1 = self.get_target_speed(r1, current_section)
        target_speed2 = self.get_target_speed(r2, current_section)
        target_speed3 = self.get_target_speed(r3, current_section)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]
        speed_data = []
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed))
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed))

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            r4 = self.get_radius([wp[self.close_index], wp[self.close_index+3], wp[self.close_index+6]])
            target_speed4 = self.get_target_speed(r4, current_section)
            speed_data.append(self.speed_for_turn(close_distance, target_speed4, current_speed))

        update = self.select_speed(speed_data)

        # self.print_speed(" -- SPEED: ", 
        #                  speed_data[0].recommended_speed_now, 
        #                  speed_data[1].recommended_speed_now, 
        #                  speed_data[2].recommended_speed_now,
        #                  (0 if len(speed_data) < 4 else speed_data[3].recommended_speed_now), 
        #                  current_speed)

        t, b = self.speed_data_to_throttle_and_brake(update)
        self.dprint("--- (" + str(cur_wp_index) + ") throt " + str(t) + " brake " + str(b) + "---")
        return t, b

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now

        # self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2)) 
        #             + " ts= " + str(round(speed_data.target_speed_at_distance, 2)) 
        #             + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))

        percent_change_per_tick = 0.07 # speed drop for one time-tick of braking
        speed_up_threshold = 0.99
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero

        if percent_of_max > 1:
            # Consider slowing down
            brake_threshold_multiplier = 1.0
            if speed_data.current_speed > 200:
                brake_threshold_multiplier = 1.0
            if percent_of_max > 1 + (brake_threshold_multiplier * percent_change_per_tick):
                if self.brake_ticks > 0:
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: counter" + str(self.brake_ticks))
                    return -1, 1
                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and not self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
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
                if self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle early2: sp_ch=" + str(percent_speed_change))
                    self.brake_ticks = 0 # done slowing down. clear brake_ticks
                    return 1, 0
                throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
                if percent_of_max > 1.02 or percent_speed_change > (-percent_change_per_tick / 2):
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle down: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain * throttle_decrease_multiple, 0 # coast, to slow down
                else:
                    # self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle maintain: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0 # done slowing down. clear brake_ticks
            # Consider speeding up
            if self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle: full speed drop: sp_ch=" + str(percent_speed_change))
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle full: p_max=" + str(percent_of_max))
                return 1, 0
            throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
            if percent_of_max < 0.98 or percent_speed_change < -0.01:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle up: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain * throttle_increase_multiple, 0 
            else:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle maintain: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain, 0

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def speed_dropping_fast(self, percent_change_per_tick: float, current_speed):
        percent_speed_change = (current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    # find speed_data with smallest recommended speed
    def select_speed(self, speed_data: [SpeedData]):
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data):
            if sd.recommended_speed_now < min_speed:
                min_speed = sd.recommended_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]
    
    def get_throttle_to_maintain_speed(self, current_speed: float):
        throttle = 0.6 + current_speed/1000
        return throttle

    def speed_for_turn(self, distance: float, target_speed: float, current_speed: float):
        d = (1/675) * (target_speed**2) + distance
        max_speed = math.sqrt(675 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def speed_for_turn_fix_physics(self, distance: float, target_speed: float, current_speed: float):
        # fix physics
        braking_decceleration = 66.0 # try 11, 14, 56
        max_speed = math.sqrt((target_speed**2) + 2 * distance * (braking_decceleration + 9.81))
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        # return a list of points with distances approximately as given 
        # in intended_target_distance[] from the current location.
        points = []
        dist = [] # for debugging
        start = roar_py_interface.RoarPyWaypoint(current_location, np.ndarray([0, 0, 0]), 0.0)
        # start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            # print("start " + str(start) + "\n- - - - -\n")
            # print("end " + str(end) +     "\n- - - - -\n")
            curr_dist += distance_p_to_p(start, end)
            # curr_dist += start.location.distance(end.location)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " +  str(dist))
        return points

    def get_radius(self, wp):
        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

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
    
    def get_target_speed(self, radius: float, current_section):
        if radius >= self.max_radius:
            return self.max_speed
        #self.section_indeces = [198, 438, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        #old section indeces = [0, 277, 554, 831, 1108, 1662, 1939, 2216, 2493]
        mu = 1.0
        if current_section == 0:
            mu = 2.8
        if current_section == 1:
            mu = 2.12
        if current_section == 2:
            mu = 1.75
        if current_section == 3:
            mu = 2.6
        if current_section == 4:
            mu = 3
        if current_section == 5:
            mu = 3
        if current_section == 6:
            mu = 2.05
        if current_section == 7:
            mu = 1.15
        if current_section == 8:
            mu = 3.4
        if current_section == 9:
            mu = 3
        if current_section == 10:
            mu = 3
        if current_section == 11:
            mu = 1.5
        if current_section == 12:
            mu = 1.35
        '''old friction coefficients (goes with old sections): 
        if current_section == 6:
            mu = 1.1
        if current_section == 7:
            mu = 1.5
        if current_section == 9:
            mu = 1.5'''
        target_speed = math.sqrt(mu*9.81*radius) * 3.6
        return max(20, min(target_speed, self.max_speed))  # clamp between 20 and max_speed

    def print_speed(self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float):
        self.dprint(text + " s1= " + str(round(s1, 2)) + " s2= " + str(round(s2, 2)) + " s3= " 
                    + str(round(s3, 2)) + " s4= " + str(round(s4, 2))
            + " cspeed= " + str(round(curr_s, 2)))

    # debug print
    def dprint(self, text):
        if self.display_debug:
            print(text)
            self.debug_strings.append(text)

# ok with mu = 2.8
NEW_WAYPOINTS = [
 new_x_y(-105.11460965268506, -726.0756666537194),
  new_x_y(-105.18821194760693, -728.0732252474694),
  new_x_y(-105.26181424252883, -730.0707838412194),
  new_x_y(-105.33541653745068, -732.0683424349694),
  new_x_y(-105.40901887376877, -734.0659021522251),
  new_x_y(-105.48262112729444, -736.0635206576256),
  new_x_y(-105.55622342221629, -738.0610792513756),
  new_x_y(-105.62982571713819, -740.0586378451256),
  new_x_y(-105.70342801206006, -742.0561964388756),
  new_x_y(-105.77703030698194, -744.0537550326256),
  new_x_y(-105.85063274233909, -746.0513174379713),
  new_x_y(-105.92422726743114, -748.0488722201256),
  new_x_y(-105.99782960374922, -750.0464319373814),
  new_x_y(-106.0714318572749, -752.0440504427819),
  new_x_y(-106.14503415219679, -754.0416090365319),
  new_x_y(-106.21863644711867, -756.0391676302819),
  new_x_y(-106.29223874204054, -758.0367262240319),
  new_x_y(-106.36584107835861, -760.0342859412876),
  new_x_y(-106.43944333188428, -762.0319044466881),
  new_x_y(-106.51304562680615, -764.0294630404381),
  new_x_y(-106.58664792172802, -766.0270216341881),
  new_x_y(-106.6602502166499, -768.0245802279381),
  new_x_y(-106.73385251157178, -770.0221388216881),
  new_x_y(-106.80745480649367, -772.0196974154381),
  new_x_y(-106.88105710141555, -774.0172560091881),
  new_x_y(-106.9546593963374, -776.0148146029381),
  new_x_y(-107.02826173265548, -778.0123743201939),
  new_x_y(-107.10186398618114, -780.0099928255944),
  new_x_y(-107.17546628110301, -782.0075514193444),
  new_x_y(-107.24906861742112, -784.0051111366001),
  new_x_y(-107.32267087094678, -786.0027296420006),
  new_x_y(-107.39627316586866, -788.0002882357506),
  new_x_y(-107.46987546079055, -789.9978468295006),
  new_x_y(-107.5434777557124, -791.9954054232506),
  new_x_y(-107.61708005063429, -793.9929640170006),
  new_x_y(-107.69068234555614, -795.9905226107506),
  new_x_y(-107.76428464047801, -797.9880812045006),
  new_x_y(-107.83788693539991, -799.9856397982506),
  new_x_y(-107.91148923032178, -801.9831983920006),
  new_x_y(-107.98509156663988, -803.9807581092564),
  new_x_y(-108.05815589596195, -805.965896376873),
  new_x_y(-108.1567581908838, -807.963454970623),
  new_x_y(-108.25536055997945, -809.9610150670634),
  new_x_y(-108.35396278072754, -811.9586331932793),
  new_x_y(-108.45256507564942, -813.9561917870293),
  new_x_y(-108.5511673705713, -815.9537503807793),
  new_x_y(-108.64976966549318, -817.9513089745293),
  new_x_y(-108.74837196041507, -819.9488675682793),
  new_x_y(-108.84697425533695, -821.9464261620293),
  new_x_y(-108.9455765502588, -823.9439847557793),
  new_x_y(-109.04417884518068, -825.9415433495293),
  new_x_y(-109.14278114010254, -827.9391019432793),
  new_x_y(-109.24138350919819, -829.9366620397196),
  new_x_y(-109.3399857299463, -831.9342801659355),
  new_x_y(-109.43858802486818, -833.9318387596855),
  new_x_y(-109.53719058178719, -835.9294026614543),
  new_x_y(-109.6357849853174, -837.9270169823418),
  new_x_y(-109.73438709239005, -839.92457177065),
  new_x_y(-109.83299739239047, -841.9221379752858),
  new_x_y(-109.93159187008303, -843.9196927635918),
  new_x_y(-110.0301941650049, -845.9172513573418),
  new_x_y(-110.12879645992679, -847.9148099510918),
  new_x_y(-110.22739875484865, -849.9123685448418),
  new_x_y(-110.32600104977053, -851.9099271385918),
  new_x_y(-110.4246033446924, -853.9074857323418),
  new_x_y(-110.52320571378803, -855.9050458287821),
  new_x_y(-110.62180793453615, -857.902663954998),
  new_x_y(-110.72041022945803, -859.900222548748),
  new_x_y(-110.81901259855367, -861.8977826451884),
  new_x_y(-110.91761481930179, -863.8954007714043),
  new_x_y(-111.01621711422365, -865.8929593651543),
  new_x_y(-111.11481940914553, -867.8905179589043),
  new_x_y(-111.2134217040674, -869.8880765526543),
  new_x_y(-111.31202399898926, -871.8856351464043),
  new_x_y(-111.41062629391115, -873.8831937401543),
  new_x_y(-111.50922858883303, -875.8807523339043),
  new_x_y(-111.6078308837549, -877.8783109276543),
  new_x_y(-111.70643317867679, -879.8758695214043),
  new_x_y(-111.80503554777242, -881.8734296178446),
  new_x_y(-111.90363776852053, -883.8710477440605),
  new_x_y(-112.0022400634424, -885.8686063378105),
  new_x_y(-112.10084243253803, -887.8661664342509),
  new_x_y(-112.19944465328615, -889.8637845604668),
  new_x_y(-112.29804694820803, -891.8613431542168),
  new_x_y(-112.3966492431299, -893.8589017479668),
  new_x_y(-112.49525153805179, -895.8564603417168),
  new_x_y(-112.59385383297365, -897.8540189354668),
  new_x_y(-112.69245612789553, -899.8515775292168),
  new_x_y(-112.79105861065219, -901.8491399284108),
  new_x_y(-112.88965308834474, -903.8466947167168),
  new_x_y(-112.98825538326662, -905.8442533104668),
  new_x_y(-113.08685775236228, -907.8418134069071),
  new_x_y(-113.18545997311037, -909.839431533123),
  new_x_y(-113.28406226803224, -911.836990126873),
  new_x_y(-113.38266463712789, -913.8345502233134),
  new_x_y(-113.48126685787598, -915.8321683495292),
  new_x_y(-113.57986915279787, -917.8297269432792),
  new_x_y(-113.67847144771974, -919.8272855370292),
  new_x_y(-113.77707374264162, -921.8248441307792),
  new_x_y(-113.87567603756351, -923.8224027245292),
  new_x_y(-113.97427833248537, -925.8199613182792),
  new_x_y(-114.07288062740724, -927.8175199120292),
  new_x_y(-114.17148292232912, -929.8150785057792),
  new_x_y(-114.27008521725098, -931.8126370995292),
  new_x_y(-114.36868758634664, -933.8101971959695),
  new_x_y(-114.46727851060454, -935.8075869994832),
  new_x_y(-114.56616076445954, -937.8016815899271),
  new_x_y(-114.67190190153678, -939.7944185430431),
  new_x_y(-114.78653512522602, -941.7866817826357),
  new_x_y(-114.91002921076856, -943.7783969367592),
  new_x_y(-115.04221598319039, -945.767232162333),
  new_x_y(-115.19063501606888, -947.7921621481603),
  new_x_y(-115.34015771598102, -949.816443466352),
  new_x_y(-115.49192817298645, -951.8399899272895),
  new_x_y(-115.64708922569675, -953.8627115295218),
  new_x_y(-115.80678230114395, -955.8845125234897),
  new_x_y(-115.97214723840844, -957.9052894792804),
  new_x_y(-116.1443220905549, -959.9249293589503),
  new_x_y(-116.32444289944362, -961.9433075940382),
  new_x_y(-116.51364343800505, -963.960286169006),
  new_x_y(-116.7130549145902, -965.9757117114635),
  new_x_y(-116.92380563403839, -967.9894135901878),
  new_x_y(-117.14702061013539, -970.0012020221217),
  new_x_y(-117.38382112417553, -972.0108661897187),
  new_x_y(-117.63532422438485, -974.0181723702251),
  new_x_y(-117.90264216101598, -976.0228620787115),
  new_x_y(-118.18688175198673, -978.0246502269283),
  new_x_y(-118.48914367400587, -980.0232233003263),
  new_x_y(-118.81052167421245, -982.0182375558761),
  new_x_y(-119.1521016974514, -984.0093172436357),
  new_x_y(-119.51496092441785, -985.9960528553356),
  new_x_y(-119.90016671602955, -987.9779994036084),
  new_x_y(-120.30877545953075, -989.954674735843),
  new_x_y(-120.74183131199504, -991.9255578870324),
  new_x_y(-121.20036483708014, -993.8900874763705),
  new_x_y(-121.68539153109593, -995.8476601527705),
  new_x_y(-122.19791023468264, -997.7976290948911),
  new_x_y(-122.73890142665677, -999.7393025716972),
  new_x_y(-123.30932539687423, -1001.6719425700168),
  new_x_y(-123.91012029528409, -1003.5947634960141),
  new_x_y(-124.54220005470222, -1005.5069309579513),
  new_x_y(-125.20645218522816, -1007.4075606380719),
  new_x_y(-125.90373543865928, -1009.2957172619053),
  new_x_y(-126.63487734172747, -1011.1704136737502),
  new_x_y(-127.40067159749782, -1013.0306100275525),
  new_x_y(-128.20187535482557, -1014.8752131028488),
  new_x_y(-129.03920634637237, -1016.7030757558811),
  new_x_y(-129.91333989633418, -1018.512996516421),
  new_x_y(-130.8249057997334, -1020.3037193412501),
  new_x_y(-131.77448507588198, -1022.0739335356295),
  new_x_y(-132.76260659942392, -1023.8222738544565),
  new_x_y(-133.7897436132254, -1025.547320795135),
  new_x_y(-134.85631012828972, -1027.2476010944845),
  new_x_y(-135.96265721684347, -1028.9215884422645),
  new_x_y(-137.10906920575704, -1030.5677044240986),
  new_x_y(-138.29575977854154, -1032.1843197067417),
  new_x_y(-139.52286799528986, -1033.769755478727),
  new_x_y(-140.79045424111365, -1035.3222851594596),
  new_x_y(-142.0984961148601, -1036.84013638979),
  new_x_y(-143.44688427117438, -1038.321493316975),
  new_x_y(-144.83541823030262, -1039.764499186739),
  new_x_y(-146.263802171403, -1041.16725925484),
  new_x_y(-147.73164072654254, -1042.527844030164),
  new_x_y(-149.2384347940055, -1043.8442928608563),
  new_x_y(-150.78357739101318, -1045.1146178743868),
  new_x_y(-152.39127986079382, -1046.3554376547681)
]


# Section 0: 319
# Section 1: 174
# Section 2: 113
# Section 3: 144
# Section 4: 124
# Section 5: 76
# Section 6: 312
# Section 7: 211
# Section 8: 252
# Section 9: 77
# Section 10: 241
# Section 11: 228
# Section 12: 129
# Section 0: 205
# Section 1: 162
# Section 2: 112
# Section 3: 145
# Section 4: 124
# Section 5: 76
# Section 6: 314
# Section 7: 212
# Section 8: 254
# Section 9: 77
# Section 10: 240
# Section 11: 228
# Section 12: 131
# Section 0: 204
# Section 1: 162
# Section 2: 112
# Section 3: 146
# Section 4: 123
# Section 5: 76
# Section 6: 316
# Section 7: 210
# Section 8: 254
# Section 9: 77
# Section 10: 240
# Section 11: 229
# Section 12: 129
# end of the loop
# done
# Solution finished in 347.90000000004505 seconds
