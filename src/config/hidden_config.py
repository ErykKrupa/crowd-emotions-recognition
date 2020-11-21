# This file should not be edited by user

from config.config import PICTURE_SIZE, KERNEL_SIZE, POOL_SIZE

PICTURE_SHAPE = (PICTURE_SIZE, PICTURE_SIZE)
INPUT_SHAPE = PICTURE_SHAPE + (3,)
KERNEL_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)
POOL_SHAPE = (POOL_SIZE, POOL_SIZE)
