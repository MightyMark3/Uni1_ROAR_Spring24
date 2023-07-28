from roar_py_interface import RoarPyActor, RoarPySensor
import typing
import gymnasium as gym

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