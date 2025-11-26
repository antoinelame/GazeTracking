import cv2
import numpy as np
import pygame
import random
import time
from gaze_tracking import GazeTracking

# 初始化Pygame
pygame.init()

# 游戏区域尺寸
GAME_WIDTH = 800
GAME_HEIGHT = 720

# 右侧显示区域尺寸
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 720

# 整体窗口尺寸
WINDOW_WIDTH = GAME_WIDTH + DISPLAY_WIDTH
WINDOW_HEIGHT = max(GAME_HEIGHT, DISPLAY_HEIGHT)

# 蛇的初始设置
SNAKE_SIZE = 20
SNAKE_SPEED = 15

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 创建窗口
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Eye Controlled Snake Game')

# 时钟
clock = pygame.time.Clock()

# 字体（使用中文支持的字体）
font_style = pygame.font.SysFont('SimHei', 50)
score_font = pygame.font.SysFont('SimHei', 35)

# 初始化GazeTracking
gaze = GazeTracking()
webcam = None
for i in range(3):
    webcam = cv2.VideoCapture(i)
    if webcam.isOpened():
        break
    else:
        webcam = None

if webcam is None:
    print("Error: Could not open webcam")
    exit()

# 游戏状态
game_state = "calibration"  # calibration, playing
control_mode = "eye"  # eye, manual

# 校准变量
calibration_start_time = None
calibration_center = (240, 240)  # 视频显示区域中心点
calibration_duration = 6  # 6秒
calibration_stable = False
calibration_buffer = []  # 用于存储最近的眼球位置

# 眼球追踪变量
gaze_center = None
dead_zone = 0.1  # 死区大小

# 蛇和食物变量
snake_list = []
length_of_snake = 1
foodx = 0
foody = 0

# 得分和时间变量
score = 0
start_time = time.time()


def our_snake(snake_size, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, GREEN, [x[0], x[1], snake_size, snake_size])


def message(msg, color, x, y):
    mesg = font_style.render(msg, True, color)
    window.blit(mesg, [x, y])


def score_display(score, speed, survival_time):
    value = score_font.render("得分: " + str(score), True, WHITE)
    window.blit(value, [10, 10])
    value = score_font.render("速度: " + str(speed), True, WHITE)
    window.blit(value, [10, 40])
    value = score_font.render("时间: " + str(int(survival_time)), True, WHITE)
    window.blit(value, [10, 70])


def generate_food():
    global foodx, foody
    foodx = round(random.randrange(0, GAME_WIDTH - SNAKE_SIZE) / 20.0) * 20.0
    foody = round(random.randrange(0, GAME_HEIGHT - SNAKE_SIZE) / 20.0) * 20.0
    # 确保食物不在蛇身上
    while [foodx, foody] in snake_list:
        foodx = round(random.randrange(0, GAME_WIDTH - SNAKE_SIZE) / 20.0) * 20.0
        foody = round(random.randrange(0, GAME_HEIGHT - SNAKE_SIZE) / 20.0) * 20.0


def main():
    global game_state, control_mode, calibration_start_time, calibration_stable, gaze_center
    global snake_list, length_of_snake, score, start_time, foodx, foody

    # 初始化蛇
    x1 = GAME_WIDTH / 2
    y1 = GAME_HEIGHT / 2
    x1_change = 0
    y1_change = 0
    snake_list = []
    length_of_snake = 1

    # 生成初始食物
    generate_food()

    # 初始化时间
    start_time = time.time()
    score = 0

    game_over = False
    game_close = False

    while not game_over:

        while game_close == True:
            window.fill(BLACK)
            message("游戏结束! 按 Q-退出 或 C-重新开始", RED, 50, 250)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        main()

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    control_mode = "manual" if control_mode == "eye" else "eye"
                if event.key == pygame.K_q:
                    game_over = True
                if control_mode == "manual":
                    if event.key == pygame.K_LEFT:
                        x1_change = -SNAKE_SIZE
                        y1_change = 0
                    elif event.key == pygame.K_RIGHT:
                        x1_change = SNAKE_SIZE
                        y1_change = 0
                    elif event.key == pygame.K_UP:
                        y1_change = -SNAKE_SIZE
                        x1_change = 0
                    elif event.key == pygame.K_DOWN:
                        y1_change = SNAKE_SIZE
                        x1_change = 0

        # 眼球控制模式
        if control_mode == "eye" and game_state == "playing":
            # 从摄像头获取帧
            _, frame = webcam.read()
            frame = cv2.flip(frame, 1)  # 镜像翻转

            # 分析帧
            gaze.refresh(frame)

            if gaze.pupils_located:
                # 获取眼球位置
                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()

                # 计算平均眼球位置
                pupil_x = (left_pupil[0] + right_pupil[0]) // 2
                pupil_y = (left_pupil[1] + right_pupil[1]) // 2

                # 计算相对于中心点的偏移
                if gaze_center is not None:
                    dx = pupil_x - gaze_center[0]
                    dy = pupil_y - gaze_center[1]

                    # 应用死区
                    if abs(dx) > dead_zone * frame.shape[1]:
                        if dx > 0:
                            x1_change = SNAKE_SIZE
                        else:
                            x1_change = -SNAKE_SIZE
                    else:
                        x1_change = 0

                    if abs(dy) > dead_zone * frame.shape[0]:
                        if dy > 0:
                            y1_change = SNAKE_SIZE
                        else:
                            y1_change = -SNAKE_SIZE
                    else:
                        y1_change = 0

        # 校准模式
        elif game_state == "calibration":
            # 从摄像头获取帧
            _, frame = webcam.read()
            frame = cv2.flip(frame, 1)  # 镜像翻转

            # 分析帧
            gaze.refresh(frame)

            if gaze.pupils_located:
                # 获取眼球位置
                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()

                # 计算平均眼球位置
                pupil_x = (left_pupil[0] + right_pupil[0]) // 2
                pupil_y = (left_pupil[1] + right_pupil[1]) // 2

                # 添加到校准缓冲区
                calibration_buffer.append((pupil_x, pupil_y))
                if len(calibration_buffer) > 30:  # 只保留最近30帧的数据
                    calibration_buffer.pop(0)

                # 检查眼球是否在中心点附近且稳定
                if len(calibration_buffer) == 30:
                    # 计算缓冲区中所有点的平均位置
                    avg_x = sum([p[0] for p in calibration_buffer]) // 30
                    avg_y = sum([p[1] for p in calibration_buffer]) // 30

                    # 检查平均位置是否在中心点附近
                    if abs(avg_x - calibration_center[0]) < 15 and abs(avg_y - calibration_center[1]) < 15:
                        if not calibration_stable:
                            calibration_start_time = time.time()
                            calibration_stable = True
                        else:
                            # 检查是否校准完成
                            if time.time() - calibration_start_time >= calibration_duration:
                                gaze_center = (avg_x, avg_y)
                                game_state = "playing"
                    else:
                        calibration_stable = False
                        calibration_start_time = None

        # 移动蛇
        x1 += x1_change
        y1 += y1_change

        # 处理边界碰撞（蛇不会死亡，而是从另一边出现）
        if x1 >= GAME_WIDTH:
            x1 = 0
        elif x1 < 0:
            x1 = GAME_WIDTH - SNAKE_SIZE
        if y1 >= GAME_HEIGHT:
            y1 = 0
        elif y1 < 0:
            y1 = GAME_HEIGHT - SNAKE_SIZE

        # 填充游戏区域
        window.fill(BLACK, rect=(0, 0, GAME_WIDTH, GAME_HEIGHT))

        # 绘制食物
        pygame.draw.rect(window, RED, [foodx, foody, SNAKE_SIZE, SNAKE_SIZE])

        # 更新蛇的位置
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        # 处理蛇身碰撞（蛇不会死亡）
        for x in snake_list[:-1]:
            if x == snake_head:
                pass  # 不做任何处理，蛇继续移动

        # 绘制蛇
        our_snake(SNAKE_SIZE, snake_list)

        # 显示得分、速度和存活时间
        survival_time = time.time() - start_time
        current_speed = SNAKE_SPEED + (length_of_snake // 5)
        score_display(score, current_speed, survival_time)

        # 显示控制模式和操作提示
        mode_text = "控制模式: 眼球" if control_mode == "eye" else "控制模式: 手动"
        mode_value = score_font.render(mode_text, True, YELLOW)
        window.blit(mode_value, [GAME_WIDTH // 2 - 100, 10])
        
        # 显示操作按钮提示
        if game_state == "playing":
            hint_text = "按 M 切换控制模式 | 按 Q 退出游戏"
            hint_value = score_font.render(hint_text, True, WHITE)
            window.blit(hint_value, [10, GAME_HEIGHT - 40])

        # 显示校准状态
        if game_state == "calibration":
            calibration_text = "正在校准... 请注视红色圆点6秒"
            calibration_value = score_font.render(calibration_text, True, YELLOW)
            window.blit(calibration_value, [10, GAME_HEIGHT - 40])

        # 处理右侧显示区域
        # 从摄像头获取帧
        _, frame = webcam.read()
        frame = cv2.flip(frame, 1)  # 镜像翻转

        # 分析帧
        gaze.refresh(frame)

        # 获取带注释的帧
        annotated_frame = gaze.annotated_frame()

        # 绘制双眼绿色十字追踪标记（线宽2像素，臂长15像素）
        if gaze.pupils_located:
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()

            # 绘制左眼球十字
            cv2.line(annotated_frame, (left_pupil[0] - 15, left_pupil[1]), (left_pupil[0] + 15, left_pupil[1]), GREEN, 2)
            cv2.line(annotated_frame, (left_pupil[0], left_pupil[1] - 15), (left_pupil[0], left_pupil[1] + 15), GREEN, 2)

            # 绘制右眼球十字
            cv2.line(annotated_frame, (right_pupil[0] - 15, right_pupil[1]), (right_pupil[0] + 15, right_pupil[1]), GREEN, 2)
            cv2.line(annotated_frame, (right_pupil[0], right_pupil[1] - 15), (right_pupil[0], right_pupil[1] + 15), GREEN, 2)

        # 绘制校准中心点
        if game_state == "calibration":
            cv2.circle(annotated_frame, calibration_center, 10, RED, -1)

        # 旋转180度
        frame_rotated = cv2.rotate(annotated_frame, cv2.ROTATE_180)
        # 调整帧大小以适应显示区域上半部分
        frame_resized = cv2.resize(frame_rotated, (360, 360))

        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 转换为Pygame表面
        frame_surface = pygame.surfarray.make_surface(frame_rgb)

        # 绘制到右侧显示区域
        window.blit(frame_surface, (GAME_WIDTH, 0))

        # 显示游戏帧数
        fps = clock.get_fps()
        fps_text = score_font.render("帧率: " + str(int(fps)), True, WHITE)
        window.blit(fps_text, (GAME_WIDTH + 10, DISPLAY_HEIGHT - 40))

        # 更新窗口
        pygame.display.update()

        # 检查是否吃到食物
        if x1 == foodx and y1 == foody:
            generate_food()
            length_of_snake += 1
            score += 10

        # 控制游戏速度
        current_speed = SNAKE_SPEED + (length_of_snake // 5)
        clock.tick(current_speed)

    # 释放资源
    webcam.release()
    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
