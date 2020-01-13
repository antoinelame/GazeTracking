from collections import deque
import math
import logging


class PointOfGaze(object):
    """
    This class tracks the user's gaze on the computer screen.
    It provides information on the position of the gaze
    in computer screen coordinates.
    """
    def __init__(self, gaze_tracking, gaze_calib, monitor, stabilize):
        """
        :param gaze_calib: a GazeCalibration object, which holds an baseline iris_size,
        and the extreme values for gaze_ratios (for each edge of the computer screen)
        :param gaze_tracking: a GazeTracking object holding detected info for the eye and iris
        :param monitor: the size of the computer monitor (in pixels)
        :param stabilize: boolean flag if EPOGs should be stabilized (True) or not (False)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)

        self.gaze_tracking = gaze_tracking
        self.gaze_calib = gaze_calib
        self.stabilize = stabilize
        self.gaze_cluster_size = 2  # three gaze points in a cluster
        self.mx_intra_cluster_variation_const = 0.3  # max variation within a cluster = 30 % of screen size

        self.mx_intra_cluster_dist = math.sqrt((self.mx_intra_cluster_variation_const * monitor['width']) ** 2
                                               + (self.mx_intra_cluster_variation_const * monitor['height']) ** 2)

        self.current_iris_size = None
        self.current_cluster = deque({}, self.gaze_cluster_size)  # max_size, when queue is full
        self.candidate_cluster = deque({}, self.gaze_cluster_size)

    def point_of_gaze(self):
        """
        Estimate the point of gaze on the computer screen based on the
        horizontal and vertical ratios.

        :return: estimated gaze x, y coordinates on the computer screen if stabilization is off
        alternatively a stabilized (clustered) coordinates, if stabilization is on
        """
        if self.current_iris_size is None:
            self.current_iris_size = self.gaze_calib.base_iris_size
            self.logger.debug('Iris {}'.format(self.current_iris_size))

        # if > 1, curr is further away compared to base, if < 1 user has moved nearer
        dist_factor = self.gaze_calib.base_iris_size / self.current_iris_size
        if self.gaze_tracking.pupils_located:
            # typical values of calibration ratios are. Note the very small diff vertically
            # gaze_calib.leftmost_hr = 0.73
            # gaze_calib.rightmost_hr = 0.48
            # gaze_calib.top_vr = 0.71
            # gaze_calib.bottom_vr = 0.88
            self.logger.debug('EPOG ratio {} {}'.
                              format(self.gaze_tracking.horizontal_ratio(), self.gaze_tracking.vertical_ratio()))
            est_x = (max(self.gaze_calib.leftmost_hr - self.gaze_tracking.horizontal_ratio(), 0) *
                     self.gaze_calib.fsw * dist_factor) /\
                    (self.gaze_calib.leftmost_hr - self.gaze_calib.rightmost_hr)
            est_y = (max(self.gaze_tracking.vertical_ratio() - self.gaze_calib.top_vr, 0) *
                     self.gaze_calib.fsh * dist_factor) /\
                    (self.gaze_calib.bottom_vr - self.gaze_calib.top_vr)
            est_x = int(round(est_x))
            est_y = int(round(est_y))

            if self.stabilize:
                stab_x, stab_y = self.stabilized(est_x, est_y)
                self.logger.debug('EPOG: raw {} {} stab {} {}'.format(est_x, est_y, stab_x, stab_y))
                est_x = stab_x
                est_y = stab_y
            else:
                self.logger.debug('EPOG raw: {} {}'.format(est_x, est_y))

                if self.looking_straight_ahead(est_x, est_y, self.gaze_calib):
                    self.current_iris_size = self.gaze_calib.measure_iris_diameter()
                    self.logger.debug('Iris {}'.format(self.current_iris_size))

            return est_x, est_y
        else:
            self.logger.debug('EPOG None None')
            return None, None

    def stabilized(self, x, y):
        """
        Stabilizes an estimated point of gaze (EPOG) by comparing it to previous EPOGs.

        :param x: estimated screen coordinate x (horizontal from left)
        :param y: estimated screen y (vertical from top)
        :return: x, y if this point is within r distance to k previously estimated gaze points
        otherwise: return the middle-point of the previous k estimates
        """
        k = self.gaze_cluster_size
        r = self.mx_intra_cluster_dist
        # bootstrap the first cluster
        if len(self.current_cluster) < k:
            self.current_cluster.append((x, y))
            return x, y
        # candidate cluster = candidate for possible eye movement coord
        # if candidate cluster is non-empty, then
        # check if dist to each member in the candidate cluster <= r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if len(self.candidate_cluster) > 0:
            if self._within_cluster(self.candidate_cluster, (x, y), r):
                self.candidate_cluster.append((x, y))
                stab_x, stab_y = self.centroid(self.candidate_cluster, x, y)
                # if the candidate cluster is big enough (size = k), then swap candidate to current
                if len(self.candidate_cluster) == k:
                    self.current_cluster = self.candidate_cluster
                    self.candidate_cluster.clear()
                return stab_x, stab_y
            # candidate cluster has no support from current point
            else:
                self.candidate_cluster.clear()
        # assert: candidate_cluster is at this point empty
        # if max dist to each member in the current cluster <= r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if self._within_cluster(self.current_cluster, (x, y), r):
            self.current_cluster.append((x, y))
        else:  # start a new candidate cluster
            self.candidate_cluster.append((x, y))
        return self.centroid(self.current_cluster, x, y)

    @staticmethod
    def centroid(cluster: deque, x, y):
        if len(cluster) > 0:
            xs = 0
            ys = 0
            for pc in cluster:
                xs = xs + pc[0]
                ys = ys + pc[1]
            return round(xs / len(cluster)), round(ys / len(cluster))
        else:
            return x, y

    @staticmethod
    def _within_cluster(c, p, r):
        """
        Check if point p is within r distance to elements in cluster c

        :param collections.deque c: the cluster
        :param p: a point (x, y)
        :param r: max distance allowed from p to any element in the cluster
        :return: True if within cluster, or if cluster is empty
        """
        for pc in c:
            dist = math.sqrt((pc[0] - p[0]) ** 2 + (pc[1] - p[1]) ** 2)
            if dist > r:
                return False
        return True

    @staticmethod
    def looking_straight_ahead(est_x, est_y, gaze_calib):
        """
        :param est_x: EPOG x coordinate
        :param est_y: EPOG y coordinate
        :param gaze_calib: (GazeCalibration) calibration object holding the screen size
        :return: True if EPOG is in the center region of the screen
        """
        wmargin = gaze_calib.fsw * 0.3
        hmargin = gaze_calib.fsh * 0.5
        wmiddle = gaze_calib.fsw / 2
        hmiddle = gaze_calib.fsh / 2
        return wmiddle - wmargin < est_x < wmiddle + wmargin and hmiddle - hmargin < est_y < hmiddle + hmargin
