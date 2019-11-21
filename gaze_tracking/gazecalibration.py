from __future__ import division
import cv2
import numpy as np
from random import randint


# TODO: calculate error (inaccuracy)
# TODO: test the effect of head movements
# This class calibrates the mapping of gaze to the screen size
# that the user is looking at.


class GazeCalibration(object):
    circle_rad: int

    def __init__(self, webcam, monitor):
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
        self.nb_test_frames = 50  # show test.py point for these many frames

        # counters to keep track of the calibration process
        self.instr_frame = 0  # counter for how many instruction frames that has been displayed
        self.calib_p = 0  # which calibration point to display
        self.fixation_frame = 0  # which fixation frame we're at for a calibration point
        self.calib_frame = 0  # how many calibration frames we're at
        self.test_p = 0  # which test.py point to display (and test.py against)
        self.test_frame = 0  # how many frames we're at

        self.calib_completed = False
        self.test_completed = False

    # Prepares nine calibration points, one at each corner of the annotated frame that is
    # displayed to the user.
    # Params:
    #       frame_size: resolution of the webcam
    #       circle_rad: radius (size) of the cbe circle that will be displayed
    def setup_calib_points(self):
        calib_points = []
        step_h = (self.fsw - 2 * self.circle_rad) // (self.nb_p[1] - 1)
        step_v = (self.fsh - 2 * self.circle_rad) // (self.nb_p[0] - 1)
        for v in range(self.nb_p[0]):
            for h in range(self.nb_p[1]):
                x = h * step_h + self.circle_rad
                y = v * step_v + self.circle_rad
                calib_points.append((x, y))
        return len(calib_points), calib_points

    # Sets up a number of random test.py points. For each point, gaze will be projected
    # onto the corresponding location on the computer screen.
    def setup_test_points(self):
        test_points = []
        minx = self.circle_rad
        maxx = self.fsw - self.circle_rad
        miny = self.circle_rad
        maxy = self.fsh - self.circle_rad
        for _ in range(self.nb_test_points):
            test_points.append((randint(minx, maxx), randint(miny, maxy)))
        return test_points

    # Display-window setup (window is full-screen and will hold the annotated calibration frame)
    @staticmethod
    def setup_calib_frame(monitor):
        fullscreen_frame = np.zeros((monitor.height, monitor.width, 3), np.uint8)
        return fullscreen_frame

    # Display a fixation circle at the fixation point for fix_nb number of frames.
    # During the last cal_nb frames, record the gaze ratios for the fixation point.
    # Calibration points are presumed to be on the borders of the screen.
    def calibrate_gaze(self, gaze):
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
            else:
                ratios = self.calib_ratios[self.calib_p]
                nb_r = len(ratios)
                hr = 0
                vr = 0
                for p in range(nb_r):
                    hr = hr + ratios[p][0]
                    vr = vr + ratios[p][1]
                self.calib_ratios[self.calib_p] = [hr / nb_r, vr / nb_r]
                self.calib_p = self.calib_p + 1
                self.fixation_frame = 0
                self.calib_frame = 0
        else:
            vmx = self.nb_p[0]
            hmx = self.nb_p[1]
            for v in range(vmx):
                for h in range(hmx):
                    i = v * vmx + h
                    if h == 0:
                        self.leftmost_hr = self.leftmost_hr + self.calib_ratios[i][0]
                    elif h == hmx - 1:
                        self.rightmost_hr = self.rightmost_hr + self.calib_ratios[i][0]
                    if v == 0:
                        self.top_vr = self.top_vr + self.calib_ratios[i][1]
                    elif v == vmx - 1:
                        self.bottom_vr = self.bottom_vr + self.calib_ratios[i][1]
            self.leftmost_hr = self.leftmost_hr / vmx
            self.rightmost_hr = self.rightmost_hr / vmx
            self.top_vr = self.top_vr / hmx
            self.bottom_vr = self.bottom_vr / hmx

            self.base_iris_size = self.base_iris_size / self.iris_size_div
            self.calib_completed = True

        return self.fs_frame

    # Display a fixation point during a number of frames
    def display_instruction(self):
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.putText(self.fs_frame, 'Please, fixate on the red dots', (200, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.7, (0, 0, 255), 1)

    # Display a fixation point during a number of frames
    def prompt_fixation(self, calib_p):
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.circle(self.fs_frame, self.calib_points[calib_p], self.circle_rad, (0, 0, 255), -1)

    # Gets the min and max from the gaze ratios that are obtained for this
    # calibration point in order to establish the screen "boundaries"
    def record_gaze_and_iris(self, calib_p, gaze):
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()
        if hr is not None and vr is not None:
            self.calib_ratios[calib_p].append([hr, vr])

        # point is in the middle of the screen: record base iris diameter (for later comparison)
        if self.calib_points[calib_p][0] == self.fsw // 2 and self.calib_points[calib_p][1] == self.fsh // 2:
            iris_diam = gaze.iris_diameter()
            self.base_iris_size = self.base_iris_size + iris_diam
            self.iris_size_div = self.iris_size_div + 1

    # Displays test.py points (red circle), and estimated gaze (blue dot) on the screen (in the frame).
    def test_gaze(self, gaze):
        self.fs_frame.fill(50)
        if self.test_p < self.nb_test_points:
            # display during nb_test frames
            if self.test_frame < self.nb_test_frames:
                # draw a red test.py circle. Params: center, rad, color, ..
                cv2.circle(self.fs_frame, self.test_points[self.test_p], self.circle_rad, (0, 0, 255), -1)
                # draw a blue marker where the gaze is estimated to be on the screen
                try:
                    (est_x, est_y) = gaze.point_of_gaze(self)
                    cv2.circle(self.fs_frame, (est_x, est_y), self.circle_rad // 4, (170, 170, 170), -1)
                except TypeError:
                    pass
                self.test_frame = self.test_frame + 1
            else:
                self.test_p = self.test_p + 1
                self.test_frame = 0
        else:
            self.test_completed = True

        return self.fs_frame

    def is_completed(self):
        return self.calib_completed

    def is_tested(self):
        return self.test_completed
