from __future__ import division

import math
import cv2
import numpy as np
from random import randint


class GazeCalibration(object):
    """
    This class calibrates the mapping of gaze to the screen size
    that the user is looking at.
    """

    def __init__(self, webcam, monitor, test_error_file):
        self.leftmost_hr = 0
        self.rightmost_hr = 0
        self.top_vr = 0
        self.bottom_vr = 0
        self.nb_p = [3, 3]  # (vert_nb_p, hor_nb_p)
        # holding the avg ratios obtained for each calibration point
        self.hr = []
        self.vr = []

        self.cam_frame_size = [int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        self.circle_rad = 20
        self.fs_frame = self.setup_calib_frame(monitor)  # make it same size as the monitor
        self.fsh, self.fsw = self.fs_frame.shape[:2]
        self.nb_calib_points, self.calib_points = self.setup_calib_points()
        self.nb_test_points = 5
        self.test_points = self.setup_test_points()

        self.calib_ratios = []
        for p in range(self.nb_calib_points):
            self.calib_ratios.append([])
        self.base_iris_size = 0
        self.iris_size_div = 0

        self.nb_instr_frames = 20  # display brief instruction on how to calibrate
        self.nb_fixation_frames = 5  # show the fixation dot for this many video frames
        self.nb_calib_frames = 10  # calibrate for this many frames (after the user has fixated on the dot)
        self.nb_test_frames = 20  # show test.py point for these many frames

        # counters to keep track of the calibration process
        self.instr_frame = 0  # counter for how many instruction frames that has been displayed
        self.calib_p = 0  # which calibration point to display
        self.fixation_frame = 0  # which fixation frame we're at for a calibration point
        self.calib_frame = 0  # how many calibration frames we're at
        self.test_p = 0  # which test.py point to display (and test.py against)
        self.test_frame = 0  # how many frames we're at

        self.test_error_file = test_error_file
        self.calib_completed = False
        self.test_completed = False

    def setup_calib_points(self):
        """
        Prepares nine calibration points, one at each corner of the annotated
        frame that is displayed to the user.

        :return: a list of calibration points (the number of points is determined
        by nb_p, which is set internally when the gaze calibration object is initialized)
        """
        calib_points = []
        step_h = (self.fsw - 2 * self.circle_rad) // (self.nb_p[1] - 1)
        step_v = (self.fsh - 2 * self.circle_rad) // (self.nb_p[0] - 1)
        for v in range(self.nb_p[0]):
            for h in range(self.nb_p[1]):
                x = h * step_h + self.circle_rad
                y = v * step_v + self.circle_rad
                calib_points.append((x, y))
        return len(calib_points), calib_points

    def setup_test_points(self):
        """
        Sets up a number of random test points. For each point, gaze will be estimated
        w.r.t. the size of the computer screen.
        """
        test_points = []
        minx = self.circle_rad
        maxx = self.fsw - self.circle_rad
        miny = self.circle_rad
        maxy = self.fsh - self.circle_rad
        for _ in range(self.nb_test_points):
            test_points.append((randint(minx, maxx), randint(miny, maxy)))
        return test_points

    @staticmethod
    def setup_calib_frame(monitor):
        """Sets up full-screen window that will hold the annotated calibration frame"""
        fullscreen_frame = np.zeros((monitor.height, monitor.width, 3), np.uint8)
        return fullscreen_frame

    def calibrate_gaze(self, gaze):
        """
        Display a fixation circle at the fixation point for fix_nb number of frames.
        During the last cal_nb frames, record the gaze ratios for the fixation point.
        Calibration points are presumed to be on the borders of the screen.
        :param gaze: a GazeTracking object (used to obtain gaze ratios for various
        calibration and test points)
        """
        self.fs_frame.fill(50)
        if self.calib_p < self.nb_calib_points:
            if self.instr_frame < self.nb_instr_frames:
                self.display_instruction()
                self.instr_frame = self.instr_frame + 1
            elif self.fixation_frame < self.nb_fixation_frames:
                self.prompt_fixation(self.calib_p)
                self.fixation_frame = self.fixation_frame + 1
            elif self.calib_frame < self.nb_calib_frames:
                self.prompt_fixation(self.calib_p)
                self.record_gaze_and_iris(self.calib_p, gaze)
                self.calib_frame = self.calib_frame + 1
            # all ratios collected for this calib point
            else:
                ratios = self.calib_ratios[self.calib_p]
                nb_r = len(ratios)
                hr = 0
                vr = 0
                for p in range(nb_r):
                    hr = hr + ratios[p][0]
                    vr = vr + ratios[p][1]
                try:
                    self.calib_ratios[self.calib_p] = [hr / nb_r, vr / nb_r]
                except ZeroDivisionError:
                    self.calib_ratios[self.calib_p] = [None, None]
                self.calib_p = self.calib_p + 1
                self.fixation_frame = 0
                self.calib_frame = 0
        # extract the avg ratio for the rightmost, etc. calibration point coordinates
        else:
            vert_nb_p = self.nb_p[0]
            hor_nb_p = self.nb_p[1]
            for v in range(vert_nb_p):
                for h in range(hor_nb_p):
                    i = v * vert_nb_p + h
                    if self.calib_ratios[i] is not None:
                        if h == 0:
                            self.leftmost_hr = self.leftmost_hr + self.calib_ratios[i][0]
                        elif h == hor_nb_p - 1:
                            self.rightmost_hr = self.rightmost_hr + self.calib_ratios[i][0]
                        if v == 0:
                            self.top_vr = self.top_vr + self.calib_ratios[i][1]
                        elif v == vert_nb_p - 1:
                            self.bottom_vr = self.bottom_vr + self.calib_ratios[i][1]
            self.leftmost_hr = self.leftmost_hr / vert_nb_p
            self.rightmost_hr = self.rightmost_hr / vert_nb_p
            self.top_vr = self.top_vr / hor_nb_p
            self.bottom_vr = self.bottom_vr / hor_nb_p

            # take the average of the recorded iris sizes
            self.base_iris_size = self.base_iris_size / self.iris_size_div
            self.calib_completed = True
        return self.fs_frame

    def display_instruction(self):
        """Display a fixation point during a number of frames"""
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.putText(self.fs_frame, 'Please, fixate on the red dots', (280, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 1.7, (0, 0, 255), 1)
        cv2.putText(self.fs_frame, 'Click on the window!', (380, 300),
                    cv2.FONT_HERSHEY_DUPLEX, 1.7, (0, 0, 255), 1)

    def prompt_fixation(self, calib_p):
        """Display a fixation point during a number of frames"""
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.circle(self.fs_frame, self.calib_points[calib_p], self.circle_rad, (0, 0, 255), -1)

    def record_gaze_and_iris(self, calib_p, gaze):
        """
        Gets the min and max from the gaze ratios that are obtained for this
        calibration point in order to establish the screen "boundaries"
        """
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()
        if hr is not None and vr is not None:
            self.calib_ratios[calib_p].append([hr, vr])

        # point is in the middle of the screen: record base iris diameter (for later comparison)
        if self.calib_points[calib_p][0] == self.fsw // 2 and self.calib_points[calib_p][1] == self.fsh // 2:
            iris_diam = gaze.measure_iris_diameter()
            self.base_iris_size = self.base_iris_size + iris_diam
            self.iris_size_div = self.iris_size_div + 1

    def test_gaze(self, gaze):
        """
        Displays test points (red circle), and estimated gaze (lightgrey smaller dots) on
        the screen (in the frame).
        """
        self.fs_frame.fill(50)
        if self.test_p < self.nb_test_points:
            # display during nb_test frames
            if self.test_frame < self.nb_test_frames:
                # draw a red test.py circle. Params: center, rad, color, ..
                cv2.circle(self.fs_frame, self.test_points[self.test_p], self.circle_rad, (0, 0, 255), -1)
                # draw a small lightgray marker where the gaze is estimated to be on the screen
                try:
                    est_x, est_y = gaze.point_of_gaze(self)
                    cv2.circle(self.fs_frame, (est_x, est_y), self.circle_rad // 4, (170, 170, 170), -1)
                    err = self.calc_error((est_x, est_y), self.test_points[self.test_p])
                    if self.test_error_file is not None:
                        self.test_error_file.write("%f\n" % err)

                except TypeError:
                    pass
                self.test_frame = self.test_frame + 1
            else:
                self.test_p = self.test_p + 1
                self.test_frame = 0
        else:
            if self.test_error_file is not None:
                self.test_error_file.close()
            self.test_completed = True

        return self.fs_frame

    @staticmethod
    def calc_error(p1, p2):
        dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        return dist

    def is_completed(self):
        return self.calib_completed

    def is_tested(self):
        return self.test_completed
