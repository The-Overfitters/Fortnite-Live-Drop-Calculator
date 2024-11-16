import time
import cv2
import numpy as np
import requests
import pyautogui
import math
import mss
import tkinter as tk
from move import move_mouse

# pyautogui.mouseInfo()
DIMENSIONS = 500


class point:
    def __init__(self, x=None, y=None, p=None):
        if p:
            self.x = p[0]
            self.y = p[1]
        elif x != None:
            self.x = x
            self.y = y
        else:
            self.y = 0
            self.x = 0

    def pair(self):
        return (self.x, self.y)


class info:
    def __init__(self, image):
        self.bus_start = point()
        self.bus_stop = point()
        self.target = point()
        self.image = image

    def get_coords(self):
        r = requests.get(
            f"https://www.landingtutorial.com/ajax/preanalyze.php?targetX={self.target.x}&targetY={self.target.y}&busStartX={self.bus_start.x}&busStartY={self.bus_start.y}&busStopX={self.bus_stop.x}&busStopY={self.bus_stop.y}"
        )
        res = list(
            map(
                lambda x: int(float(x) * DIMENSIONS),
                r.text[5:-7].split(","),
            )
        )

        return [point(res[0], res[1]), point(res[2], res[3]), res[4]]

    def draw(self):
        cv2.circle(
            self.image, (self.target.x, self.target.y), 10, (0, 100, 100), thickness=10
        )

    def show(self):
        cv2.imshow("map", cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)


def find_marker(image, marker, method=cv2.TM_CCOEFF):
    w, h = marker.shape[::-1]
    res = cv2.matchTemplate(image, marker, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    bottom_mid = (top_left[0] + w // 2, top_left[1] + h)
    adjusted_bottom_mid = (
        (top_left[0] + w // 2) * 500 / DIMENSIONS,
        (top_left[1] + h) * 500 / DIMENSIONS,
    )

    return bottom_mid, adjusted_bottom_mid


def shot(
    dimensions={
        "top": 90,
        "left": -1460,
        "width": 1000,
        "height": 970,
    }
):
    with mss.mss() as sct:
        monitor = dimensions
        image = np.array(sct.grab(monitor))
        return image


import numpy as np


def find_angle(j, g, maxi):
    print(j, maxi, g)
    g = np.subtract(g, j)
    j = np.subtract(j, maxi)

    print(j, g)
    dot = np.dot(j, g)
    return np.rad2deg(np.arccos(dot / (magnitude(j) * magnitude(g))))


def magnitude(v):
    return math.sqrt(sum(pow(element, 2) for element in v))


def drop_angle(slope, jump, glide):

    path_angle = -slope * 45

    turn = find_angle(jump, glide)

    return path_angle - turn


def test_templates(marker, img_2):
    methods = [
        "TM_CCOEFF",
        "TM_CCOEFF_NORMED",
        "TM_CCORR",
        "TM_CCORR_NORMED",
        "TM_SQDIFF",
        "TM_SQDIFF_NORMED",
    ]
    w, h = marker.shape[::-1]
    temp_img = img_2.copy()
    for meth in methods:
        img_2 = temp_img.copy()
        method = getattr(cv2, meth)

        # Apply template Matching
        res = cv2.matchTemplate(img_2, marker, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img_2, top_left, bottom_right, 255, 2)
        cv2.imshow(meth, img_2)
        cv2.waitKey(0)


def save_map():
    image = shot()
    image = cv2.resize(image, (500, 500))
    cv2.imshow("shot", image)
    cv2.waitKey(0)
    cv2.imwrite("savedmap.png", image)


def threshold_bus(image):

    thresh = cv2.inRange(image, (93, 70, 55), (94, 80, 81))
    points = np.where(thresh == 255)

    return thresh, points


def check_start(image):
    thresh = cv2.inRange(image, (150, 50, 50), (250, 90, 90))
    points = np.where(thresh == 255)
    return thresh, points


def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def main():
    load = True
    if load:
        image = cv2.imread("basesmall3.png")
    else:
        image = shot()
        image = cv2.resize(image, (500, 500))
        cv2.rectangle(image, (440, 270), (500, 290), (0, 0, 0), 20)

    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    thresh = cv2.inRange(image, (0, 205, 190), (60, 255, 255))

    points = np.where(thresh == 255)
    mini = [points[1][0], points[0][0]]
    maxi = [points[1][-1], points[0][-1]]
    slope = (mini[1] - maxi[1]) / (maxi[0] - mini[0])
    p = [[], []]

    while len(p[0]) < 1:
        if not load:
            image = shot()
            image = cv2.resize(image, (500, 500))
            cv2.rectangle(image, (440, 270), (500, 290), (0, 0, 0), 20)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        t, p = threshold_bus(image)
        print(p)

    # cv2.imshow("thresh", t)
    # cv2.waitKey(0)
    # exit()

    mini_point = point(mini[0], mini[1])
    maxi_point = point(maxi[0], maxi[1])
    p_point = point(p[1][0], p[0][0])
    cv2.circle(image, p_point.pair(), 5, (255, 0, 0), 5)
    if distance(mini_point, p_point) < distance(maxi_point, p_point):
        start = mini
        end = maxi
    else:
        start = maxi
        end = mini

    # standardize to 500 scale if size != 500

    # adj_mini = (int(mini[0] * 500 / DIMENSIONS), int(mini[1] * 500 / DIMENSIONS))
    # adj_mini = (int(mini[0] * 500 / DIMENSIONS), int(mini[1] * 500 / DIMENSIONS))
    # adj_maxi = (int(maxi[0] * 500 / DIMENSIONS), int(maxi[1] * 500 / DIMENSIONS))
    # adj_maxi = (int(maxi[0] * 500 / DIMENSIONS), int(maxi[1] * 500 / DIMENSIONS))

    marker = cv2.imread("marker.png", cv2.IMREAD_GRAYSCALE)

    img_2 = image.copy()

    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Apply template Matching

    bottom, adj_bottom = find_marker(img_2, marker)

    cv2.circle(image, bottom, 5, (255, 0, 0), 5)

    cv2.circle(image, start, 2, (0, 255, 0), 5)

    cv2.circle(image, end, 2, (255, 0, 0), 5)

    cv2.line(image, mini, maxi, (255, 0, 0), 3)

    map_info = info(image)
    map_info.draw()

    map_info.bus_start = point(*start)
    map_info.bus_stop = point(*end)
    map_info.target = point(*adj_bottom)

    [jump, glide, _] = map_info.get_coords()

    cv2.circle(image, (jump.x, jump.y), 3, (255, 0, 0), thickness=5)
    cv2.circle(image, (glide.x, glide.y), 3, (0, 0, 255), thickness=5)
    angle = find_angle([jump.x, jump.y], [glide.x, glide.y], maxi)
    print(angle)
    print(math.atan(600 / (math.dist(jump.pair(), glide.pair()))) * 180 / math.pi)
    if not load:
        move_mouse(angle * (9970 / 360), 0)
        move_mouse(
            0, math.atan(600 / (math.dist(jump.pair(), glide.pair()))) * (9970 / 360)
        )
    return map_info


if __name__ == "__main__":
    testing = True

    if testing:
        m = main()
        m.show()
        cv2.waitKey(0)
    else:
        while True:
            pic = shot({"top": 150, "left": -200, "width": 50, "height": 50})
            pic = cv2.cvtColor(pic, cv2.COLOR_BGRA2RGB)

            t, p = check_start(pic)
            cv2.imshow("thresh", t)
            cv2.waitKey(1)
            if len(p[0]) > 0:
                pyautogui.press("m")
                time.sleep(3)
                cv2.destroyAllWindows()
                time.sleep(3)
                m = main()
                m.show()
                break
