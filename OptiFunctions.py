import PIL.Image
import PIL.ImageOps
import array
import sys
import numpy as np
import math
import cairo
import time
import cv2
import pytweening
import gi
import threading

gi.require_version('LightDM', '1')
gi.require_version('Gtk', '3.0')


from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from gi.repository import GObject, LightDM
import os
from PIL import ImageEnhance
RES_PATH = os.getcwd() + "/res/"
DEV = None

STATES = ["INIT",
          "AUTH_D1",
          "AUTH_D2",
          "LOGIN"]

STATE_LABELS = ["Authenticating user...",
                "User found in network!",
                "User found in network!",
                "Please Login"]

def set_resource_path():
    global RES_PATH
    RES_PATH = os.getcwd() + "/res/"


class ProcTimer:
    def __init__(self):
        self.beg = time.time()
        self.end = None
        self.name = "NA"

    def start(self, name):
        self.beg = time.time()
        self.name = name

    def stop(self):
        self.end = time.time()
        timed = self.end - self.beg
        print("Process " + self.name + " took: " + str(timed))


TIMER = ProcTimer()


class Camera:
    def __init__(self, rate,window_width):
        self.cam_open = False
        self.cam = None
        for i in range(3):
            self.cam = cv2.VideoCapture(i)
            if self.cam is not None and self.cam.isOpened():
                self.cam_open = True
                break
        self.refresh_rate = rate
        self.refresh_counter = 0
        self.masked_img_buffer = None
        self.last_window_width = None
        self.image_size = None
        self.curr_image = self.cam.read()
        self.get_masked_img(window_width)

    def get_masked_img(self, window_width):
        self.curr_image = self.cam.read()

        self.refresh_counter += 1

        if self.last_window_width is None:
            self.last_window_width = window_width

        if self.refresh_counter >= self.refresh_rate:
            self.refresh_counter = 0

        if self.refresh_counter == 0 or self.masked_img_buffer is None:
            if self.cam_open:
             #   TIMER.start("Read Img")
                retval, cam_image = self.curr_image
            #    retval, cam_image = self.cam.retrieve()
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
                cam_image = PIL.Image.fromarray(cam_image).convert('RGB')
                cam_image = PIL.ImageOps.mirror(cam_image)
              #  TIMER.stop()
            else:
                cam_image = PIL.Image.open(RES_PATH + "no_cam.png").convert('RGB')

            if cam_image.width > cam_image.height:
                fin_dim = cam_image.height
                dim_cropped = cam_image.width - cam_image.height
                cam_image = cam_image.crop((dim_cropped / 2, 0, fin_dim + dim_cropped / 2, fin_dim))
            else:
                fin_dim = cam_image.width
                cam_image = cam_image.crop((0, 0, fin_dim, fin_dim))
            cam_image = cam_image.crop((0, 0, fin_dim, fin_dim))

            fin_size = int(window_width * 0.3)
            cam_image = cam_image.resize((fin_size, fin_size), PIL.Image.NEAREST)
            cam_image = ImageEnhance.Contrast(cam_image).enhance(0.95)
            RGB = np.array(cam_image)
            h, w = RGB.shape[:2]
            RGBA = np.dstack((RGB, mat_mask(w))).astype('uint8')

            mBlack = (RGBA[:, :, 0:3] == [0, 0, 0]).all(2)
            RGBA[mBlack] = (0, 0, 0, 0)
            final_image = PIL.Image.fromarray(RGBA)
            self.image_size = final_image.size
            ims = pil2cairo(PIL.Image.fromarray(RGBA))
            self.masked_img_buffer = ims
            #self.cam.grab()

    def set_refresh_rate(self, rate):
        self.refresh_rate = rate

    def release(self):
        self.cam.release()

class State:
    def __init__(self, window, info_label):
        self.state = 0
        self.label = info_label
        self.set_state(STATES[0])
        self.state_int = 0
        self.tick = 0
        self.init_time = time.time()
        self.prev_time = time.time()

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.reset_time()
        self.state = state
        if self.label is not None:
            set_info_label(self.label, STATE_LABELS[STATES.index(state)])
        self.tick = 0
        self.init_time = time.time()
        self.prev_time = time.time()

    def get_time(self):
        return self.tick

    def inc_time(self):
        self.tick += time.time()-self.prev_time
        self.prev_time = time.time()

    def reset_time(self):
        self.tick = 0
        self.init_time = time.time()
        self.prev_time = time.time()

    def inc_state(self):
        self.reset_time()
        self.state_int += 1
        if self.state_int >= len(STATES):
            self.state_int = 0
        self.set_state(STATES[self.state_int])


class Animation:
    def __init__(self, time_per_step, loop, eq):
        self.completion = 0
        self.tween = 0
        self.time_per_step = time_per_step
        self.equation = eq
        self.complete = False
        self.looping = loop
        self.last_time = time.time()
        self.curr_time = time.time()
        self.delta_time = 0
        self.paused = False

    def step(self):
        if self.complete is False:
            self.curr_time = time.time()
            self.delta_time = self.curr_time - self.last_time
            self.last_time = time.time()
            if self.paused is False:
                next_comp = self.completion + (self.delta_time*self.time_per_step)
                if next_comp >= 1:
                    next_comp = 1
                self.tween = self.equation(next_comp)
                self.completion = next_comp
                if self.completion >= 1:
                    if not self.looping:
                        self.finish()
                    else:
                        self.completion = 0
        return self.tween

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def finish(self):
        self.completion = 1.0
        self.complete = True

    def restart(self):
        self.complete = False
        self.completion = 0

    def is_finished(self):
        return self.complete

def get_masked_img(src, arc, draw):
    mask = PIL.Image.new('L', (src.width,src.height), 0)
    draw.pieslice((0, 0) + mask.size,0,arc, fill=255)
    src = PIL.ImageOps.fit(src, mask.size, centering=(0.5, 0.5))
    src.putalpha(mask)
    return src


def pil2cairo(im):
    # NOTE This function is not my own, it is a workaround designed by
    # https://stackoverflow.com/users/141253/joaquin-cuenca-abela with
    # some minor modifications by myself.

    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    s = im.tobytes('raw', 'BGRA')
    a = array.array('B', s)
    dest = cairo.ImageSurface(cairo.FORMAT_ARGB32, im.size[0], im.size[1])
    ctx = cairo.Context(dest)
    non_premult_src_wo_alpha = cairo.ImageSurface.create_for_data(
        a, cairo.FORMAT_RGB24, im.size[0], im.size[1])
    non_premult_src_alpha = cairo.ImageSurface.create_for_data(
        a, cairo.FORMAT_ARGB32, im.size[0], im.size[1])
    ctx.set_source_surface(non_premult_src_wo_alpha)
    ctx.mask_surface(non_premult_src_alpha)

    return dest


def mat_mask(n):
    cent = int(n / 2)
    y, x = np.ogrid[-cent:n - cent, -cent:n - cent]
    mask = x ** 2 + y ** 2 <= cent * cent
    mat_arr = np.zeros((n, n))
    mat_arr[mask] = 255
    return mat_arr


def image2pixbuf(im):
    arr = array.array('B', im.tobytes())
    width, height = im.size
    return GdkPixbuf.Pixbuf.new_from_data(arr, GdkPixbuf.Colorspace.RGB,True, 8, width, height, width * 4)


def set_info_label(label, text):
    markup = "<span font_desc='Source Code Pro Bold "
    size = 20
    markup = markup + str(size) +"'>" + text + "</span>"
    label.set_markup(markup)
