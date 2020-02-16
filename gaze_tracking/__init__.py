from .gaze_tracking import GazeTracking
from .gazecalibration import GazeCalibration
from .iriscalibration import IrisCalibration
from .point_of_gaze import PointOfGaze
from .screensize import get_screensize
from .epog import EPOG
import logging.config

# To turn off logging, edit logging.conf, and set root logging level to CRTICIAL
# so that only critical messages will be displayed
logging.config.fileConfig('logging.conf')
