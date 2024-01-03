#from src.data_modules import impuluse_response
import os, sys

sys.path.append(os.getcwd())

from src.data_modules.impulse_response import ImpulseResponseDataModule

ImpulseResponseDataModule()