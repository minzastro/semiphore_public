import gpustat
import numpy as np


def meminfo(logger):
    info = gpustat.GPUStatCollection.new_query().gpus[0]
    logger.debug("GPU memory consimption: %s MB", info.memory_used)
    return info.memory_used


def get_total_gpu_memory():
    info = gpustat.GPUStatCollection.new_query().gpus[0]
    return info.memory_total


def get_size_for_cuda(num_seds, num_bands, num_redshifts=50):
    return 8 * (3 + 2 * num_bands +
                num_redshifts * (2 + 2 * num_seds + num_bands)) / (1024**2)


def get_batch_and_grid(sed_count, mag_count, redshifts):
    batch_size = int(0.75 * get_total_gpu_memory() /
                     get_size_for_cuda(sed_count, mag_count, redshifts))
    block_size = 512
    grid_size = 2**min(15, int(np.ceil(np.log2(batch_size) * 0.5)))
    batch_size = min(block_size * grid_size, batch_size)
    return batch_size, grid_size, block_size
