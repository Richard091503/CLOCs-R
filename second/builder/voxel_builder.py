import numpy as np

from spconv.utils import VoxelGeneratorV2
from second.protos import voxel_generator_pb2


class VoxelGenerator(VoxelGeneratorV2):
    """spconv 1.2 compatibility: VoxelGeneratorV2 (no full_mean, no block
    filtering, same as spconv v1.0) returning the v1.0 3-tuple."""
    def generate(self, points, max_voxels=None):
        res = super().generate(points, max_voxels)
        return res["voxels"], res["coordinates"], res["num_points_per_voxel"]


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_config.voxel_size),
        point_cloud_range=list(voxel_config.point_cloud_range),
        max_num_points=voxel_config.max_number_of_points_per_voxel,
        max_voxels=20000)
    return voxel_generator
