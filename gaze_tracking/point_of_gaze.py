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

        self.gaze_tracking = gaze_tracking
        self.gaze_calib = gaze_calib
        self.stabilize = stabilize
        self.nb_same = 2  # nb of move fragments in the same direction in order to be an eye movement
        self.nb_interv = 0  # nb of intervening in diff direction within a smooth pursuit

        self.gaze_cluster_size = 2  # three gaze points in a cluster
        self.mx_intra_cluster_variation_const = 0.3  # max variation within a cluster = 30 % of screen size
        self.mx_intra_cluster_dist = self.mx_intra_cluster_variation_const * monitor['width']
        self.current_iris_size = None
        self.ongoing_eye_move = False
        self.candidate_eye_move = False
        self.cluster_mn_size = 2   # min cluster size
        self.cluster_mx_size = 20  # max cluster size
        self.mx_intra_cluster_variation_const = 0.3  # max variation within a cluster = 30 % of screen size
        self.mx_intra_cluster_dist = self.mx_intra_cluster_variation_const * monitor['width']
        self.current_cluster_x = deque({}, self.cluster_mx_size)  # max_size, when queue is full
        self.current_cluster_y = deque({}, self.cluster_mx_size)  # max_size, when queue is full
        self.candidate_cluster_x = deque({}, self.cluster_mx_size)
        self.candidate_cluster_y = deque({}, self.cluster_mx_size)
        self.move_cluster_x = deque({}, self.cluster_mx_size)
        self.move_cluster_y = deque({}, self.cluster_mx_size)

    def point_of_gaze(self, webcam_estate):
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
            # typical values of calibration ratios are. Note the very small vertical diff
            # gaze_calib.leftmost_hr = 0.73
            # gaze_calib.rightmost_hr = 0.48
            # gaze_calib.top_vr = 0.71
            # gaze_calib.bottom_vr = 0.88
            # self.logger.debug('EPOG ratio {} {}'.
            #                   format(self.gaze_tracking.horizontal_ratio(), self.gaze_tracking.vertical_ratio()))
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
                    self.current_iris_size = self.gaze_calib.measure_iris_diameter(webcam_estate)
                    self.logger.debug('Iris {}'.format(self.current_iris_size))

            return est_x, est_y
        else:
            # self.logger.debug('EPOG None None')
            return None, None

    def stabilized(self, x, y):
        """
        Stabilizes an estimated point of gaze (EPOG) by comparing it to previous EPOGs.

        :param x: estimated screen coordinate x (horizontally from left)
        :param y: estimated screen y (vertically from top)
        :return: x, y if this point is within r distance to k previously estimated gaze points
        otherwise: return the middle-point of the previous k estimates
        """
        r = self.mx_intra_cluster_dist
        # candidate cluster = candidate for possible eye movement coord
        # if candidate cluster is non-empty, then
        # check if dist of new point to each older member in the candidate cluster <= r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if self.candidate_eye_move or self.ongoing_eye_move:
            self.move_cluster_x.append(x)
            self.move_cluster_y.append(y)
            if self.eye_movement(self.current_cluster_x, self.move_cluster_x, self.nb_same):
                self.ongoing_eye_move = True
                self.candidate_eye_move = False
                self.candidate_cluster_x.clear()
                self.candidate_cluster_y.clear()
                return x, y
            # no current support for eye movement
            else:
                self.ongoing_eye_move = False
                if self.candidate_eye_move:
                    self.candidate_eye_move = False
                else:
                    self.current_cluster_x.clear()
                    self.current_cluster_y.clear()
                self.move_cluster_x.clear()
                self.move_cluster_y.clear()

        if len(self.candidate_cluster_x) > 0:
            if self._within_cluster(self.candidate_cluster_x, x, r):
                self.candidate_cluster_x.append(x)
                self.candidate_cluster_y.append(y)
                self.current_cluster_x.clear()
                self.current_cluster_y.clear()
                stab_x = self.mean(self.candidate_cluster_x)
                stab_y = self.mean(self.candidate_cluster_y)
                # if the candidate cluster is big enough (size = k), then swap candidate to current
                if len(self.candidate_cluster_x) == self.cluster_mn_size:
                    self.current_cluster_x = deque(self.candidate_cluster_x, self.cluster_mx_size)
                    self.current_cluster_y = deque(self.candidate_cluster_y, self.cluster_mx_size)
                    self.candidate_cluster_x.clear()
                    self.candidate_cluster_y.clear()
                return stab_x, stab_y
            # candidate cluster has no support from current point
            else:
                self.candidate_cluster_x.clear()
                self.candidate_cluster_y.clear()

        # assert: move_cluster and candidate_cluster are at this point empty
        # if max dist to each member in the current cluster <= r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if self._within_cluster(self.current_cluster_x, x, r):
            self.current_cluster_x.append(x)
            self.current_cluster_y.append(y)
        else:  # start a new move and candidate cluster, future data will tell which is correct
            self.candidate_eye_move = True
            self.move_cluster_x.append(x)
            self.move_cluster_y.append(y)

            self.candidate_cluster_x.append(x)
            self.candidate_cluster_y.append(y)

        stab_x = self.mean(self.current_cluster_x)
        stab_y = self.mean(self.current_cluster_y)
        return stab_x, stab_y

    @staticmethod
    def mean(cluster: deque):
        if len(cluster) > 0:
            xs = 0
            for x in cluster:
                xs = xs + x
            return round(xs / len(cluster))
        else:
            return None, None

    @staticmethod
    def _within_cluster(c, x, r):
        """
        Check if value x is within r distance to elements in cluster c

        :param collections.deque c: the cluster
        :param x: a value
        :param r: max distance allowed from p to any element in the cluster
        :return: True if within cluster, or if cluster is empty
        """
        for x2 in c:
            if abs(x2 - x) > r:
                return False
        return True

    def eye_movement(self, curr_xs, move_xs, nb_same):
        xs = []
        xs.extend(curr_xs)
        xs.extend(move_xs)
        # check if pairs of consecutive diffs are in the same direction
        nb_interv = 0
        if nb_same > 1:
            # print('xs: ', xs, 'ys: ', ys)
            if len(xs) > nb_same:
                for d in range(nb_same - 1):
                    # determine angle between two vectors
                    xd1 = xs[-d - 1] - xs[-d - 2]
                    xd2 = xs[-d - 2] - xs[-d - 3]
                    if xd1 * xd2 <= 0:
                        if nb_interv < self.nb_interv:
                            nb_interv = nb_interv + 1
                        else:
                            return False
                # print(self.xs, self.ys)
                return True
            else:
                return False
        else:
            return False

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
