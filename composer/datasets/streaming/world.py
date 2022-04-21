from torch.utils.data import get_worker_info

from composer.utils import dist


class World(object):
    """Context about workers, devices, and nodes. Useful for dividing work.

    Fields:
      - node / num_nodes

      - global_device / global_num_devices
      - device_of_node / devices_per_node

      - global_worker / global_num_workers
      - worker_of_node / workers_per_node
      - worker_of_device / workers_per_device
    """

    def __init__(self, global_worker: int, num_nodes: int, devices_per_node: int, workers_per_device: int) -> None:
        """Initialize with enough information to populate fields.

        Args:
            global_worker (int): Global worker ID (across devices and nodes).
            num_nodes (int): Global number of nodes in this training job.
            devices_per_node (int): Number of devices on each node.
            workers_per_device (int): Number of workers per each device.
        """
        workers_per_node = workers_per_device * devices_per_node
        global_num_devices = devices_per_node * num_nodes
        global_num_workers = workers_per_node * num_nodes

        worker_of_node = global_worker % workers_per_node
        worker_of_device = global_worker % workers_per_device

        global_device = global_worker // workers_per_device
        device_of_node = global_device % devices_per_node

        node = global_device // devices_per_node

        self.node = node
        self.num_nodes = num_nodes

        self.global_device = global_device
        self.global_num_devices = global_num_devices
        self.device_of_node = device_of_node
        self.devices_per_node = devices_per_node

        self.global_worker = global_worker
        self.global_num_workers = global_num_workers
        self.worker_of_node = worker_of_node
        self.workers_per_node = workers_per_node
        self.worker_of_device = worker_of_device
        self.workers_per_device = workers_per_device

    @classmethod
    def from_env(cls):
        """Initialize from environment and get_worker_info()."""
        info = get_worker_info()
        if info:
            worker_of_device = info.id
            workers_per_device = info.num_workers
        else:
            worker_of_device = 0
            workers_per_device = 1

        devices_per_node = dist.get_local_world_size()
        workers_per_node = workers_per_device * devices_per_node

        global_device = dist.get_global_rank()
        device_of_node = global_device % devices_per_node
        # worker_of_node = device_of_node * workers_per_device + worker_of_device

        global_num_devices = dist.get_world_size()
        assert not global_num_devices % devices_per_node
        num_nodes = global_num_devices // devices_per_node
        # global_num_workers = workers_per_device * global_num_devices

        node = dist.get_node_rank()
        global_worker = node * workers_per_node + device_of_node * workers_per_device + worker_of_device

        return cls(global_worker, num_nodes, devices_per_node, workers_per_device)
