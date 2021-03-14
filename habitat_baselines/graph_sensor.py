import habitat
import numpy as np
# from torch_geometric.data import Data

from typing import Any
from gym import spaces

# import habitat.utils.visualizations.maps as maps
from PIL import Image

def graph_update(graph):
    # print("################################################### GRAPH UPDATE ###################################################")
    graph[0] = True
    return graph


@habitat.registry.register_sensor(name="edges_sensor")
class EdgesSensor(habitat.Sensor):
    cls_uuid: str = "edges"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self.sim = sim

        # print('################################################## EDGES SENSOR ##################################################')
        # map = maps.get_topdown_map_from_sim(self.sim, meters_per_pixel=0.1)
        # print('########################## SIM AGENT_0 HEIGHT:',
        #       self.sim.get_agent(0).state.position[1])
        # print(map)
        # # print(self.sim.get_agent_state().position)
        # im = Image.fromarray(map)
        # im.save("/project/test_map.jpeg")
        #
        # map = maps.get_topdown_map(self.sim.pathfinder, height=0,  meters_per_pixel=0.1)
        # print('########################## SIM MAP HEIGHT 0:', map)
        # im = Image.fromarray(map)
        # im.save("/project/test_map_height_0.jpeg")

        self.graph = self.config.GRAPH
        self.shape_ = np.array(self.graph[2]).shape

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
                shape=self.shape_,
                dtype=np.float32,
                )

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        # print('################################################## EDGES SENSOR OBSERVATION ##################################################')
        if self.graph[0]:
            self.graph[0] = False
        else:
            graph_update(self.graph)

        edges = self.graph[2]
        return np.array(edges)


@habitat.registry.register_sensor(name="nodes_sensor")
class NodesSensor(habitat.Sensor):
    cls_uuid: str = "nodes"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        # print('################################################## NODES SENSOR ##################################################')
        self.graph = self.config.GRAPH
        self.shape_ = np.array(self.graph[1]).shape

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
            shape=self.shape_,
            dtype=np.float32,
        )

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        # print('################################################## NODES SENSOR OBSERVATION ##################################################')
        if self.graph[0]:
            self.graph[0] = False
        else:
            graph_update(self.graph)
        nodes = self.graph[1]
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


@habitat.registry.register_sensor(name="metric_map_sensor")
class MetricMapSensor(habitat.Sensor):
    cls_uuid: str = "metric_map"

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        print('################################################## METRIC MAP SENSOR ##################################################')
        self.metric_map = self.config.METRIC_MAP
        self.shape_ = self.metric_map.shape

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.COLOR

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self.shape_,
            dtype=np.float32,
        )

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        print('################################################## METRIC MAP SENSOR OBSERVATION ##################################################')
        return self.metric_map
