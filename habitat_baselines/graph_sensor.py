import habitat
import numpy as np
from torch_geometric.data import Data

from typing import Any
from gym import spaces

@habitat.registry.register_sensor(name="edges_sensor")
class EdgesSensor(habitat.Sensor):
    cls_uuid: str = "edges"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self.graph = self.config.GRAPH

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.NORMAL

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,4),
                dtype=np.float32,
                )


    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        edges = [[0, 1, 1, 2], [1, 0, 2, 1]]
        return np.array(edges)


@habitat.registry.register_sensor(name="nodes_sensor")
class NodesSensor(habitat.Sensor):
    cls_uuid: str = "nodes"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self.graph = self.config.GRAPH

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.NORMAL

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3, 1),
            dtype=np.float32,
        )

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        nodes = [[-1], [0], [1]]
        return np.array(nodes)


@habitat.registry.register_sensor(name="graph_sensor")
class GraphSensor(habitat.Sensor):
    cls_uuid: str = "graph"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        print('################################################## GRAPH SENSOR ##################################################')

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.NORMAL

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        # return spaces.Dict({
        #     'edge_index': spaces.Box(
        #         low=np.finfo(np.float32).min,
        #         high=np.finfo(np.float32).max,
        #         shape=(2,4),
        #         dtype=np.float32,
        #         ),
        #     'x': spaces.Box(
        #         low=np.finfo(np.float32).min,
        #         high=np.finfo(np.float32).max,
        #         shape=(3,1),
        #         dtype=np.float32,
        #     )
        # })
        return spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,4),
                dtype=np.float32,
                )


    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        print(
            "################################################## GRAPH SENSOR DOES SOMETHING ##################################################")
        # edge_index = torch.tensor([[0, 1, 1, 2],
        #                            [1, 0, 2, 1]], dtype=torch.long)
        # x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        #
        # data = Data(x=x, edge_index=edge_index)

        # data = {
        #     'edge_index': [[0, 1, 1, 2], [1, 0, 2, 1]],
        #     'x': [[-1], [0], [1]]
        # }
        # data = [
        #     [[0, 1, 1, 2], [1, 0, 2, 1]],
        #     [[-1], [0], [1]]
        # ]
        data = [[0, 1, 1, 2], [1, 0, 2, 1]]
        return np.array(data)
