from roar_py_interface import RoarPyActor, RoarPySensor
import math
import typing
import gymnasium as gym
import roar_py_interface
import pygame
from PIL.Image import Image
import numpy as np
from typing import Optional, Dict, Any

class ManualControlViewer:
    def __init__(
        self
    ):
        self.screen = None
        self.clock = None
        self.last_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }
        self._font_mono = None
    
    def init_pygame(self, x, y) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((x, y), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("RoarPy Manual Control Viewer")
        pygame.key.set_repeat()
        self.clock = pygame.time.Clock()
        self.init_font()

    def init_font(self):
        font_name = "courier"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)

    def close(self) -> None:
        pygame.quit()

    def render(self, image : roar_py_interface.RoarPyCameraSensorData, occupancy_map : Optional[Image] = None, vehicle = None) -> Optional[Dict[str, Any]]:
        image_pil : Image = image.get_image()
        occupancy_map_rgb = occupancy_map.convert("RGB") if occupancy_map is not None else None
        if self.screen is None:
            if occupancy_map_rgb is None:
                self.init_pygame(image_pil.width, image_pil.height)
            else:
                self.init_pygame(image_pil.width + occupancy_map.width, image_pil.height)
        
        new_control = {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.0,
            "hand_brake": np.array([0]),
            "reverse": np.array([0])
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[pygame.K_UP]:
            new_control['throttle'] = 0.4
        if pressed_keys[pygame.K_DOWN]:
            new_control['brake'] = 0.2
        if pressed_keys[pygame.K_LEFT]:
            new_control['steer'] = -0.2
        if pressed_keys[pygame.K_RIGHT]:
            new_control['steer'] = 0.2
        
        image_surface = pygame.image.fromstring(image_pil.tobytes(), image_pil.size, image_pil.mode).convert()
        if occupancy_map_rgb is not None:
            occupancy_map_surface = pygame.image.fromstring(occupancy_map_rgb.tobytes(), occupancy_map_rgb.size, occupancy_map_rgb.mode).convert()

        self.screen.fill((0,0,0))
        self.screen.blit(image_surface, (0, 0))
        if occupancy_map_rgb is not None:
            self.screen.blit(occupancy_map_surface, (image_pil.width, 0))

        if vehicle is not None:
            info = self.get_info(vehicle)
            self.show_info(info)
            # info_surface = pygame.Surface((320, 100))
            # info_surface.set_alpha(100)
            # self.screen.blit(info_surface, (0, 0))
            # v_offset = 4
            # msg = "Location: " + str(vehicle.get_3d_location())
            # surface = self._font_mono.render(msg, True, (255, 255, 255))
            # self.screen.blit(surface, (8, v_offset))
            # v_offset += 18

        pygame.display.flip()
        self.clock.tick(60)
        self.last_control = new_control
        return new_control
    
    def get_info(self, vehicle):
        location = vehicle.get_3d_location()
        v = vehicle.get_linear_3d_velocity()
        return [
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (location[0], location[1])),
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)),
        ]

    def show_info(self, string_list):
        info_surface = pygame.Surface((320, 100))
        info_surface.set_alpha(100)
        self.screen.blit(info_surface, (0, 0))
        v_offset = 4
        for msg in string_list:
            surface = self._font_mono.render(msg, True, (255, 255, 255))
            self.screen.blit(surface, (8, v_offset))
            v_offset += 18


class RoarCompetitionAgentWrapper(RoarPyActor):
    def __init__(self, wrapped : RoarPyActor):
        self._wrapped = wrapped
    
    @property
    def control_timestep(self) -> float:
        return self._wrapped.control_timestep
    
    @property
    def force_real_control_timestep(self) -> bool:
        return self._wrapped.force_real_control_timestep

    def get_sensors(self) -> typing.Iterable[RoarPySensor]:
        return self._wrapped.get_sensors()

    def get_action_spec(self) -> gym.Space:
        return self._wrapped.get_action_spec()
    
    async def _apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped._apply_action(action)

    def close(self):
        pass

    def is_closed(self) -> bool:
        return self._wrapped.is_closed()

    def __del__(self):
        pass
    
    async def apply_action(self, action: typing.Any) -> bool:
        return await self._wrapped.apply_action(action)

    def get_gym_observation_spec(self) -> gym.Space:
        return self._wrapped.get_gym_observation_spec()

    async def receive_observation(self) -> typing.Dict[str, typing.Any]:
        return await self._wrapped.receive_observation()
    
    def get_last_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_observation()
    
    def get_last_gym_observation(self) -> typing.Optional[typing.Dict[str,typing.Any]]:
        return self._wrapped.get_last_gym_observation()

    def convert_obs_to_gym_obs(self, observation : typing.Dict[str,typing.Any]) -> typing.Dict[str,typing.Any]:
        return self._wrapped.convert_obs_to_gym_obs(observation)
