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
          return np.array([0.48, 0.05, 0.02]) # 

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
        mu = 1.0 #TODO: set mu for each section
        if current_section == 0:
            mu = 2.8
        if current_section == 1:
            mu = 2.15
        if current_section == 2:
            mu = 1.85
        if current_section == 3:
            mu = 2.2
        if current_section == 4:
            mu = 2.2
        if current_section == 5:
            mu = 2.5
        if current_section == 6:
            mu = 2.0
        if current_section == 7:
            mu = 1.2
        if current_section == 8:
            mu = 3.6
        if current_section == 9:
            mu = 3.6
        if current_section == 10:
            mu = 2.3
        if current_section == 11:
            mu = 1.5
        if current_section == 12:
            mu = 1.4
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
new_x_y(-105.61528778076172, -726.1124877929688),
  new_x_y(-105.68889007568359, -728.1100463867188),
  new_x_y(-105.76249237060549, -730.1076049804688),
  new_x_y(-105.83609466552734, -732.1051635742188),
  new_x_y(-105.90969696044922, -734.1027221679688),
  new_x_y(-105.9832992553711, -736.100341796875),
  new_x_y(-106.05690155029295, -738.097900390625),
  new_x_y(-106.13050384521485, -740.095458984375),
  new_x_y(-106.20410614013672, -742.093017578125),
  new_x_y(-106.2777084350586, -744.090576171875),
  new_x_y(-106.35131072998048, -746.088134765625),
  new_x_y(-106.42490539550779, -748.085693359375),
  new_x_y(-106.49850769042968, -750.083251953125),
  new_x_y(-106.57210998535156, -752.0808715820312),
  new_x_y(-106.64571228027344, -754.0784301757812),
  new_x_y(-106.71931457519533, -756.0759887695312),
  new_x_y(-106.7929168701172, -758.0735473632812),
  new_x_y(-106.86651916503907, -760.0711059570312),
  new_x_y(-106.94012145996093, -762.0687255859375),
  new_x_y(-107.0137237548828, -764.0662841796875),
  new_x_y(-107.08732604980467, -766.0638427734375),
  new_x_y(-107.16092834472656, -768.0614013671875),
  new_x_y(-107.23453063964844, -770.0589599609375),
  new_x_y(-107.30813293457032, -772.0565185546875),
  new_x_y(-107.38173522949221, -774.0540771484375),
  new_x_y(-107.45533752441406, -776.0516357421875),
  new_x_y(-107.52893981933593, -778.0491943359375),
  new_x_y(-107.6025421142578, -780.0468139648438),
  new_x_y(-107.67614440917967, -782.0443725585938),
  new_x_y(-107.74974670410157, -784.0419311523438),
  new_x_y(-107.82334899902344, -786.03955078125),
  new_x_y(-107.89695129394532, -788.037109375),
  new_x_y(-107.9705535888672, -790.03466796875),
  new_x_y(-108.04415588378906, -792.0322265625),
  new_x_y(-108.11775817871094, -794.02978515625),
  new_x_y(-108.1913604736328, -796.02734375),
  new_x_y(-108.26496276855467, -798.02490234375),
  new_x_y(-108.33856506347657, -800.0224609375),
  new_x_y(-108.41216735839843, -802.02001953125),
  new_x_y(-108.48576965332033, -804.017578125),
# ---
  new_x_y(-107.0593719482422, -806.0151977539062),
  new_x_y(-107.15797424316406, -808.0127563476562),
  new_x_y(-107.25657653808594, -810.0103149414062),
  new_x_y(-107.3551788330078, -812.0079345703125),
  new_x_y(-107.45378112792967, -814.0054931640625),
  new_x_y(-107.55238342285156, -816.0030517578125),
  new_x_y(-107.65098571777344, -818.0006103515625),
  new_x_y(-107.74958801269533, -819.9981689453125),
  new_x_y(-107.8481903076172, -821.9957275390625),
  new_x_y(-107.94679260253906, -823.9932861328125),
  new_x_y(-108.04539489746094, -825.9908447265625),
  new_x_y(-108.1439971923828, -827.9884033203125),
  new_x_y(-108.24259948730467, -829.9859619140625),
  new_x_y(-108.34120178222656, -831.9835815429688),
  new_x_y(-108.43980407714844, -833.9811401367188),
  new_x_y(-108.53840637207033, -835.9786987304688),
  new_x_y(-108.63700103759766, -837.976318359375),
  new_x_y(-108.73560333251952, -839.973876953125),
  new_x_y(-108.83421325683594, -841.971435546875),
  new_x_y(-108.93280792236328, -843.968994140625),
  new_x_y(-109.03141021728516, -845.966552734375),
  new_x_y(-109.13001251220705, -847.964111328125),
  new_x_y(-109.2286148071289, -849.961669921875),
  new_x_y(-109.32721710205078, -851.959228515625),
  new_x_y(-109.42581939697266, -853.956787109375),
  new_x_y(-109.52442169189452, -855.954345703125),
  new_x_y(-109.6230239868164, -857.9519653320312),
  new_x_y(-109.72162628173828, -859.9495239257812),
  new_x_y(-109.82022857666016, -861.9470825195312),
  new_x_y(-109.91883087158205, -863.9447021484375),
  new_x_y(-110.0174331665039, -865.9422607421875),
  new_x_y(-110.11603546142578, -867.9398193359375),
  new_x_y(-110.21463775634766, -869.9373779296875),
  new_x_y(-110.31324005126952, -871.9349365234375),
  new_x_y(-110.4118423461914, -873.9324951171875),
  new_x_y(-110.51044464111328, -875.9300537109375),
  new_x_y(-110.60904693603516, -877.9276123046875),
  new_x_y(-110.70764923095705, -879.9251708984375),
  new_x_y(-110.8062515258789, -881.9227294921875),
  new_x_y(-110.90485382080078, -883.9203491210938),
  new_x_y(-111.00345611572266, -885.9179077148438),
  new_x_y(-111.10205841064452, -887.9154663085938),
  new_x_y(-111.2006607055664, -889.9130859375),
  new_x_y(-111.29926300048828, -891.91064453125),
  new_x_y(-111.39786529541016, -893.908203125),
  new_x_y(-111.49646759033205, -895.90576171875),
  new_x_y(-111.5950698852539, -897.9033203125),
  new_x_y(-111.69367218017578, -899.90087890625),
  new_x_y(-111.79227447509766, -901.8984375),
  new_x_y(-111.890869140625, -903.89599609375),
  new_x_y(-111.98947143554688, -905.8935546875),
  new_x_y(-112.08807373046876, -907.89111328125),
  new_x_y(-112.18667602539062, -909.8887329101562),
  new_x_y(-112.2852783203125, -911.8862915039062),
  new_x_y(-112.38388061523438, -913.8838500976562),
  new_x_y(-112.48248291015624, -915.8814697265624),
  new_x_y(-112.58108520507812, -917.8790283203124),
  new_x_y(-112.6796875, -919.8765869140624),
  new_x_y(-112.77828979492188, -921.8741455078124),
  new_x_y(-112.87689208984376, -923.8717041015624),
  new_x_y(-112.97549438476562, -925.8692626953124),
  new_x_y(-113.0740966796875, -927.8668212890624),
  new_x_y(-113.17269897460938, -929.8643798828124),
  new_x_y(-113.27130126953124, -931.8619384765624),
  new_x_y(-113.36990356445312, -933.8594970703124),
  new_x_y(-113.468505859375, -935.8571166992188),
  new_x_y(-113.56756591796876, -937.8546752929688),
  new_x_y(-113.67355346679688, -939.8518676757812),
  new_x_y(-113.7884521484375, -941.8485717773438),
  new_x_y(-113.9122314453125, -943.8447265625),
# ---
  new_x_y(-114.04489135742188, -945.84033203125),
  new_x_y(-114.19335191541585, -947.8658263670991),
  new_x_y(-114.34295862358141, -949.8912363143883),
  new_x_y(-114.49485748782062, -951.9164755388969),
  new_x_y(-114.65019421999166, -953.9414538154948),
  new_x_y(-114.8101140771268, -955.9660750837393),
  new_x_y(-114.97576168414288, -957.9902355047961),
  new_x_y(-115.14828083455009, -960.0138215202265),
  new_x_y(-115.32881426367257, -962.0367079132702),
  new_x_y(-115.51850338890462, -964.0587558733644),
  new_x_y(-115.71848801154046, -966.0798110647662),
  new_x_y(-115.92990597473293, -968.0997017003018),
  new_x_y(-116.15389277215913, -970.1182366214357),
  new_x_y(-116.39158110199925, -972.1352033860504),
  new_x_y(-116.64410036086976, -974.1503663655434),
  new_x_y(-116.91257607239395, -976.1634648530819),
  new_x_y(-117.1981292451454, -978.1742111851191),
  new_x_y(-117.50187565476051, -980.182288878549),
  new_x_y(-117.82492504509014, -982.1873507861768),
  new_x_y(-118.16838024334682, -984.1890172735032),
  new_x_y(-118.53333618430507, -986.1868744201508),
  new_x_y(-118.92087883873, -988.1804722496252),
  new_x_y(-119.33208404134551, -990.1693229914691),
  new_x_y(-119.76801621380903, -992.1528993802644),
  new_x_y(-120.22972697833833, -994.1306329963394),
  new_x_y(-120.71825365783725, -996.1019126534635),
  new_x_y(-121.23461765859643, -998.0660828392442),
  new_x_y(-121.77982273190084, -1000.0224422143918),
  new_x_y(-122.35485311116318, -1001.9702421774747),
  new_x_y(-122.96067152152226, -1003.9086855022548),
  new_x_y(-123.5982170591993, -1005.8369250551733),
  new_x_y(-124.26840293829741, -1007.7540626010284),
  new_x_y(-124.9721141031604, -1009.6591477053787),
  new_x_y(-125.71020470487949, -1011.5511767426849),
  new_x_y(-126.48349544105352, -1013.4290920196802),
  new_x_y(-127.29277075846966, -1015.2917810239428),
  new_x_y(-128.1387759189826, -1017.1380758081002),
  new_x_y(-129.02221392952967, -1018.966752520554),
  new_x_y(-129.94374233793062, -1020.776531094048),
  new_x_y(-130.90396989688583, -1022.5660751038163),
  new_x_y(-131.90345309940554, -1024.3339918074405),
  new_x_y(-132.94269258977798, -1026.0788323789002),
  new_x_y(-134.02212945511505, -1027.79909234963),
  new_x_y(-135.14214140350444, -1029.4932122696725),
  new_x_y(-136.30303883584224, -1031.1595786022606),
  new_x_y(-137.50506081952557, -1032.7965248653372),
  new_x_y(-138.74837097334472, -1034.4023330336545),
  new_x_y(-140.03305327413284, -1035.9752352151427),
  new_x_y(-141.35910779700293, -1037.5134156152353),
  new_x_y(-142.72644640232664, -1039.015012802736),
  new_x_y(-144.1348883839853, -1040.478122290638),
  new_x_y(-145.58415609484518, -1041.900799445027),
  new_x_y(-147.07387056687514, -1043.281062734825),
  new_x_y(-148.60354714482904, -1044.6168973346482),
  new_x_y(-150.17259115395154, -1045.9062590924402),
  new_x_y(-151.78029362373218, -1047.1470788728216)
]

# NEW_WAYPOINTS = [
#   new_x_y(-110.03141021728516, -845.966552734375),
#   new_x_y(-110.11751251220704, -847.964111328125),
#   new_x_y(-110.2036148071289, -849.961669921875),
#   new_x_y(-110.28971710205079, -851.959228515625),
#   new_x_y(-110.37581939697266, -853.956787109375),
#   new_x_y(-110.46192169189452, -855.954345703125),
#   new_x_y(-110.5480239868164, -857.9519653320312),
#   new_x_y(-110.63412628173828, -859.9495239257812),
#   new_x_y(-110.72022857666016, -861.9470825195312),
#   new_x_y(-110.80633087158205, -863.9447021484375),
#   new_x_y(-110.8924331665039, -865.9422607421875),
#   new_x_y(-110.97853546142578, -867.9398193359375),
#   new_x_y(-111.06463775634765, -869.9373779296875),
#   new_x_y(-111.15074005126952, -871.9349365234375),
#   new_x_y(-111.23684234619141, -873.9324951171875),
#   new_x_y(-111.32294464111328, -875.9300537109375),
#   new_x_y(-111.40904693603515, -877.9276123046875),
#   new_x_y(-111.49514923095704, -879.9251708984375),
#   new_x_y(-111.58125152587891, -881.9227294921875),
#   new_x_y(-111.66735382080078, -883.9203491210938),
#   new_x_y(-111.75345611572266, -885.9179077148438),
#   new_x_y(-111.83955841064451, -887.9154663085938),
#   new_x_y(-111.9256607055664, -889.9130859375),
#   new_x_y(-112.01176300048829, -891.91064453125),
#   new_x_y(-112.09786529541016, -893.908203125),
#   new_x_y(-112.18396759033205, -895.90576171875),
#   new_x_y(-112.2700698852539, -897.9033203125),
#   new_x_y(-112.35617218017578, -899.90087890625),
#   new_x_y(-112.44227447509766, -901.8984375),
#   new_x_y(-112.528369140625, -903.89599609375),
#   new_x_y(-112.61447143554688, -905.8935546875),
#   new_x_y(-112.70057373046876, -907.89111328125),
#   new_x_y(-112.78667602539062, -909.8887329101562),
#   new_x_y(-112.8727783203125, -911.8862915039062),
#   new_x_y(-112.95888061523438, -913.8838500976562),
#   new_x_y(-113.04498291015624, -915.8814697265624),
#   new_x_y(-113.13108520507812, -917.8790283203124),
#   new_x_y(-113.2171875, -919.8765869140624),
#   new_x_y(-113.30328979492188, -921.8741455078124),
#   new_x_y(-113.38939208984377, -923.8717041015624),
#   new_x_y(-113.47549438476562, -925.8692626953124),
# # ---
#   new_x_y(-113.5740966796875, -927.8668212890624),
#   new_x_y(-113.67269897460938, -929.8643798828124),
#   new_x_y(-113.77130126953124, -931.8619384765624),
#   new_x_y(-113.86990356445312, -933.8594970703124),
#   new_x_y(-113.968505859375, -935.8571166992188),
#   new_x_y(-114.06756591796876, -937.8546752929688),
#   new_x_y(-114.17355346679688, -939.8518676757812),
#   new_x_y(-114.2884521484375, -941.8485717773438),
#   new_x_y(-114.4122314453125, -943.8447265625),
# # ---
#   new_x_y(-114.54489135742188, -945.84033203125),
#   new_x_y(-114.69335191541585, -947.8658263670991),
#   new_x_y(-114.84295862358141, -949.8912363143883),
#   new_x_y(-114.99485748782062, -951.9164755388969),
#   new_x_y(-115.15019421999166, -953.9414538154948),
#   new_x_y(-115.3101140771268, -955.9660750837393),
#   new_x_y(-115.47576168414288, -957.9902355047961),
#   new_x_y(-115.64828083455009, -960.0138215202265),
#   new_x_y(-115.82881426367257, -962.0367079132702),
#   new_x_y(-116.01850338890462, -964.0587558733644),
#   new_x_y(-116.21848801154046, -966.0798110647662),
#   new_x_y(-116.42990597473293, -968.0997017003018),
#   new_x_y(-116.65389277215913, -970.1182366214357),
#   new_x_y(-116.89158110199925, -972.1352033860504),
#   new_x_y(-117.14410036086976, -974.1503663655434),
#   new_x_y(-117.41257607239395, -976.1634648530819),
#   new_x_y(-117.6981292451454, -978.1742111851191),
#   new_x_y(-118.00187565476051, -980.182288878549),
#   new_x_y(-118.32492504509014, -982.1873507861768),
#   new_x_y(-118.66838024334682, -984.1890172735032),
#   new_x_y(-119.03333618430507, -986.1868744201508),
#   new_x_y(-119.42087883873, -988.1804722496252),
#   new_x_y(-119.83208404134551, -990.1693229914691),
#   new_x_y(-120.26801621380903, -992.1528993802644),
#   new_x_y(-120.72972697833833, -994.1306329963394),
#   new_x_y(-121.21825365783725, -996.1019126534635),
#   new_x_y(-121.73461765859643, -998.0660828392442),
#   new_x_y(-122.27982273190084, -1000.0224422143918),
#   new_x_y(-122.85485311116318, -1001.9702421774747),
#   new_x_y(-123.46067152152226, -1003.9086855022548),
#   new_x_y(-124.0982170591993, -1005.8369250551733),
#   new_x_y(-124.76840293829741, -1007.7540626010284),
#   new_x_y(-125.4721141031604, -1009.6591477053787),
#   new_x_y(-126.21020470487949, -1011.5511767426849),
#   new_x_y(-126.98349544105352, -1013.4290920196802),
#   new_x_y(-127.79277075846966, -1015.2917810239428),
#   new_x_y(-128.6387759189826, -1017.1380758081002),
#   new_x_y(-129.52221392952967, -1018.966752520554),
#   new_x_y(-130.44374233793062, -1020.776531094048),
#   new_x_y(-131.40396989688583, -1022.5660751038163),
#   new_x_y(-132.40345309940554, -1024.3339918074405),
#   new_x_y(-133.44269258977798, -1026.0788323789002),
#   new_x_y(-134.52212945511505, -1027.79909234963),
#   new_x_y(-135.64214140350444, -1029.4932122696725),
#   new_x_y(-136.80303883584224, -1031.1595786022606),
#   new_x_y(-138.00506081952557, -1032.7965248653372),
#   new_x_y(-139.24837097334472, -1034.4023330336545),
#   new_x_y(-140.53305327413284, -1035.9752352151427),
#   new_x_y(-141.85910779700293, -1037.5134156152353),
#   new_x_y(-143.22644640232664, -1039.015012802736),
#   new_x_y(-144.6348883839853, -1040.478122290638),
#   new_x_y(-146.08415609484518, -1041.900799445027),
#   new_x_y(-147.57387056687514, -1043.281062734825),
#   new_x_y(-149.10354714482904, -1044.6168973346482),
#   new_x_y(-150.67259115395154, -1045.9062590924402),
#   new_x_y(-152.28029362373218, -1047.1470788728216)
# ]
# NEW_WAYPOINTS = [
#   new_x_y(-136.57873095686156, -1030.203535606267),
#   new_x_y(-137.8667972204935, -1031.7598115981493),
#   new_x_y(-139.18252347896276, -1033.2927724595745),
#   new_x_y(-140.52611614234127, -1034.801367221037),
#   new_x_y(-141.8977418181508, -1036.2845182910253),
#   new_x_y(-143.29752547677396, -1037.7411220029483),
#   new_x_y(-144.725548584683, -1039.1700492661232),
#   new_x_y(-146.18184720885785, -1040.570146325282),
#   new_x_y(-147.66641009612138, -1041.9402356330686),
#   new_x_y(-149.1791767314901, -1043.27911683998),
#   new_x_y(-150.72003538001985, -1044.5855679061715),
#   new_x_y(-152.2888211170207, -1045.8583463394914),
#   new_x_y(-153.88531385192078, -1047.0961905640343),
#   new_x_y(-155.50923635147518, -1048.2978214234038),
#   new_x_y(-157.16025226844073, -1049.4619438227433),
#   new_x_y(-158.8379641822742, -1050.5872485134437),
#   new_x_y(-160.54191165885163, -1051.672414024255),
#   new_x_y(-162.27156933665685, -1052.716108742313)
# ]