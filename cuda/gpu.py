import gpustat
import numpy as np

"""
GPU-specific routines.
"""

def meminfo(logger):
    info = gpustat.GPUStatCollection.new_query().gpus[0]
    logger.debug("GPU memory consimption: %s MB", info.memory_used)
    return info.memory_used


def get_total_gpu_memory():
    info = gpustat.GPUStatCollection.new_query().gpus[0]
    return info.memory_total


def get_size_for_photoz(num_seds, num_bands, num_redshifts=50):
    return 8 * (3 + 2 * num_bands +
                num_redshifts * (2 + 2 * num_seds + num_bands)) / (1024**2)


def get_photoz_gpu_dims(sed_count, mag_count, redshifts):
    batch_size = int(0.75 * get_total_gpu_memory() /
                     get_size_for_photoz(sed_count, mag_count, redshifts))
    block_size = 512
    grid_size = 2**min(15, int(np.ceil(np.log2(batch_size) * 0.5)))
    batch_size = min(block_size * grid_size, batch_size)
    return batch_size, grid_size, block_size


def get_size_for_fit(sed_count, mag_count):
    return 16 * (sed_count + mag_count + mag_count * sed_count)


def get_fit_gpu_dims(sed_count, mag_count, max_batch_size=256**2):
    batch_size = int(0.75 * get_total_gpu_memory() * 1024**2 /
                     get_size_for_fit(sed_count, mag_count))
    batch_size = min(batch_size, max_batch_size)
    block_size = 512
    grid_size = 2**min(15, int(np.ceil(np.log2(batch_size) * 0.5)))
    batch_size = min(block_size * grid_size, batch_size)
    return batch_size, grid_size, block_size
