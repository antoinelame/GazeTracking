from __future__ import division
import cv2
from random import randint


# This class calibrates the mapping of gaze to the screen size
# that the user is looking at.
class GazeCalibration(object):
    circle_rad: int

    def __init__(self, webcam):
        self.leftmost_hr = 0
        self.rightmost_hr = 0
        self.top_vr = 0
        self.bottom_vr = 0

        self.leftmost_hr_div = 0
        self.rightmost_hr_div = 0
        self.top_vr_div = 0
        self.bottom_vr_div = 0

        self.frame_size = [int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        self.circle_rad = 20
        self.nb_calib_points, self.calib_points = self.setup_calib_points()
        self.nb_test_points = 5
        self.test_points = self.setup_test_points()
        self.window_name = self.setup_calib_window()
        # TODO: obtain actual screen size (instead of screen resolution chosen by cv2
        #  when displaying a "fullscreen" window).
        self.ww, self.wh = self.get_screen_size()

        self.nb_instr_frames = 20  # display brief instruction on how to calibrate
        self.nb_fixation_frames = 15  # show the fixation dot for this many video frames
        self.nb_calib_frames = 10  # calibrate for this many frames (after the user has fixated on the dot)
        self.nb_test_frames = 20  # show test.py point for these many frames
        # counters to keep track of the calibration process
        self.instr_frame = 0  # counter for how many instruction frames that has been displayed
        self.calib_p = 0  # which calibration point to display
        self.fixation_frame = 0  # which fixation frame we're at for a calibration point
        self.calib_frame = 0  # how many calibration frames we're at
        self.test_p = 0  # which test.py point to display (and test.py against)
        self.test_frame = 0  # how many frames we're at

        self.calib_completed = False
        self.test_completed = False

    # Prepares four calibration points, one at each corner of the annotated frame that is
    # displayed to the user.
    # Params:
    #       frame_size: resolution of the webcam
    #       circle_rad: radius (size) of the cbe circle that will be displayed
    def setup_calib_points(self):
        calib_points = [(self.circle_rad, self.circle_rad),
                        (self.frame_size[0] - self.circle_rad, self.circle_rad),
                        (self.frame_size[0] - self.circle_rad, self.frame_size[1] - self.circle_rad),
                        (self.circle_rad, self.frame_size[1] - self.circle_rad)]
        return len(calib_points), calib_points

    # Sets up a number of random test.py points. For each point, gaze will be projected
    # onto the corresponding location on the computer screen.
    def setup_test_points(self):
        test_points = []
        minx = self.circle_rad
        maxx = self.frame_size[0] - self.circle_rad
        miny = self.circle_rad
        maxy = self.frame_size[1] - self.circle_rad
        for _ in range(self.nb_test_points):
            test_points.append((randint(minx, maxx), randint(miny, maxy)))
        return test_points

    # Display-window setup (window is full-screen and will hold the annotated calibration frame)
    @staticmethod
    def setup_calib_window():
        cv2.namedWindow('Calibration', cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return 'Calibration'

    # Screen size is estimated through the current cv2 fullscreen window resolution.
    def get_screen_size(self):
        xx, xy, ww, wh = cv2.getWindowImageRect(self.window_name)
        # xx, xy are grey bars surrounding the window. We compensate for this, to get the
        # full screen size. Note that this size is what cv2 has chosen to display the webcam
        # in "fullscreen". The actual screen resolution can be higher.
        return ww + 2 * xx, wh + 2 * xy

    # Display a fixation circle at the fixation point for fix_nb number of frames.
    # During the last cal_nb frames, record the gaze ratios for the fixation point.
    # Calibration points are presumed to be on the borders of the screen.
    def calibrate_gaze(self, frame, gaze):
        if self.calib_p < self.nb_calib_points:
            if self.instr_frame < self.nb_instr_frames:
                self.display_instruction(frame)
                self.instr_frame = self.instr_frame + 1
            elif self.fixation_frame < self.nb_fixation_frames:
                self.prompt_fixation(self.calib_p, frame)
                self.fixation_frame = self.fixation_frame + 1
            elif self.calib_frame < self.nb_calib_frames:
                self.prompt_fixation(self.calib_p, frame)
                self.record_gaze_ratio(self.calib_p, gaze)
                self.calib_frame = self.calib_frame + 1
            else:
                self.calib_p = self.calib_p + 1
                self.fixation_frame = 0
                self.calib_frame = 0
        else:
            self.leftmost_hr = self.leftmost_hr / self.leftmost_hr_div
            self.leftmost_hr_div = 1
            self.rightmost_hr = self.rightmost_hr / self.rightmost_hr_div
            self.rightmost_hr_div = 1
            self.top_vr = self.top_vr / self.top_vr_div
            self.top_vr_div = 1
            self.bottom_vr = self.bottom_vr / self.bottom_vr_div
            self.top_vr_div = 1
            self.calib_completed = True

    # Display a fixation point during a number of frames
    @staticmethod
    def display_instruction(frame):
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.putText(frame, 'Please, fixate on the red dots', (200, 50), cv2.FONT_HERSHEY_DUPLEX, 1.7, (0, 0, 255), 1)

    # Display a fixation point during a number of frames
    def prompt_fixation(self, calib_p, frame):
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.circle(frame, self.calib_points[calib_p], self.circle_rad, (0, 0, 255), -1)

    # Gets the min and max from the gaze ratios that are obtained for this
    # calibration point in order to establish the screen "boundaries"
    def record_gaze_ratio(self, calib_p, gaze):
        margin = 10
        if gaze.horizontal_ratio() is not None:
            if self.calib_points[calib_p][0] - self.circle_rad <= margin:  # left border
                self.leftmost_hr = self.leftmost_hr + gaze.horizontal_ratio()
                self.leftmost_hr_div = self.leftmost_hr_div + 1
            elif self.calib_points[calib_p][0] + self.circle_rad >= self.frame_size[0] - margin:  # right border
                self.rightmost_hr = self.rightmost_hr + gaze.horizontal_ratio()
                self.rightmost_hr_div = self.rightmost_hr_div + 1
        if gaze.vertical_ratio() is not None:
            if self.calib_points[calib_p][1] - self.circle_rad <= margin:  # top border
                self.top_vr = self.top_vr + gaze.vertical_ratio()
                self.top_vr_div = self.top_vr_div + 1
            elif self.calib_points[calib_p][1] + self.circle_rad >= self.frame_size[1] - margin:  # bottom border
                self.bottom_vr = self.bottom_vr + gaze.vertical_ratio()
                self.bottom_vr_div = self.bottom_vr_div + 1

    # Displays test.py points (red circle), and estimated gaze (blue dot) on the screen (in the frame).
    def test_gaze(self, frame, gaze):
        if self.test_p < self.nb_test_points:
            # display during nb_test frames
            if self.test_frame < self.nb_test_frames:
                # draw a red test.py circle. Params: center, rad, color, ..
                cv2.circle(frame, self.test_points[self.test_p], self.circle_rad, (0, 0, 255), -1)
                # draw a blue marker where the gaze is estimated to be on the screen
                try:
                    (est_x, est_y) = gaze.point_of_gaze(self, self.frame_size)
                    cv2.circle(frame, (est_x, est_y), self.circle_rad, (255, 0, 0), -1)
                except TypeError:
                    pass
                self.test_frame = self.test_frame + 1
            else:
                self.test_p = self.test_p + 1
                self.test_frame = 0
        else:
            self.test_completed = True

    def is_calib_completed(self):
        return self.calib_completed

    def is_test_completed(self):
        return self.test_completed
