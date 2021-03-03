import habitat

@habitat.registry.register_sensor(name="graph_sensor")
class GraphSensor(habitat.Sensor):
    def __init__(self, config):
        super().__init__(config=config)
        print('################################################## GRAPH SENSOR ##################################################')

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self) -> str:
        return "graph"

    # Defines the type of the sensor
    # def _get_sensor_type(self, *args: Any, **kwargs: Any):
    #     return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    # def _get_observation_space(self, *args: Any, **kwargs: Any):
    #     return spaces.Box(
    #         low=np.finfo(np.float32).min,
    #         high=np.finfo(np.float32).max,
    #         shape=(3,),
    #         dtype=np.float32,
    #     )

    # This is called whenever reset is called or an action is taken
    def get_observation(
        self, observations
    ):
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        return data
