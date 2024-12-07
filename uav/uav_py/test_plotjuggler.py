#!/usr/bin/env python

from __future__ import annotations

__package__ = "chdrft.projects.uav_py"
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field
import pydantic
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import chdrft.utils.Z as Z
from chdrft.dsp.image import ImageData
from chdrft.config.env import g_env
import enum
import struct
import crcmod
import ctypes
import time
import threading
import queue
from chdrft.tube.connection import Connection
import contextlib
import requests
import datetime
import concurrent.futures
from chdrft.display.render import ImageGrid
from chdrft.inputs.controller import OGamepad, configure_joy, debug_controller_state
from chdrft.utils.rx_helpers import ImageIO, rx, pipe_connect
from chdrft.dsp.utils import linearize_clamp
from .lib.siyi import *
from .lib.proto import comm_p2p, comm_pb2
import subprocess as sp
from pydantic_core import from_json

if not g_env.slim:
  from chdrft.display.service import g_plot_service as oplt

import Pyro5 as pyro
import Pyro5.server, Pyro5.nameserver, Pyro5.client

import zmq

import rich.json
import rich.live

global flags, cache
flags = None
cache = None

#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def args(parser):
  clist = CmdsList()
  parser.add_argument('--folder')
  parser.add_argument('--feed-rate', type=float, default=1)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class SimplePubSub(Node):

  def __init__(self):
    super().__init__('simple_pub_sub')

    topic_name = 'video_frames'

    self.publisher_ = self.create_publisher(Image, topic_name, 10)
    #self.timer = self.create_timer(0.1, self.timer_callback)

    #self.cap = cv2.VideoCapture(0)
    self.br = CvBridge()

    self.subscription = self.create_subscription(Image, topic_name, self.img_callback, 10)
    self.subscription
    self.br = CvBridge()

  def timer_callback(self):
    ret, frame = self.cap.read()
    if ret == True:
      self.publisher_.publish(self.br.cv2_to_imgmsg(frame))
    self.get_logger().info('Publishing video frame')

  def img_callback(self, data):

    self.get_logger().info('Receiving video frame')
    current_frame = self.br.imgmsg_to_cv2(data)
    #cv2.imshow("camera", current_frame)




def test_pub(ctx):
  cmisc.return_to_ipython()
  #%%
  from rclpy_message_converter import message_converter
  dt = SiyiJoyData()
  print(message_converter.convert_dictionary_to_ros_message(dt.dict()))
  #%%

def test_image_pub(ctx):
  #%%

  #%%

  tg = tunnel.ThreadGroup()

  async def proc():
    await asyncio.sleep(0.1)
    simple_pub_sub = SimplePubSub()
    files =cmisc.get_input('./data/101SIYI_IMG/*.jpg')
    for f in files:
      i = ImageData.Make(f)
      
      print('FEEDING')
      simple_pub_sub.publisher_.publish(simple_pub_sub.br.cv2_to_imgmsg(i.norm_cv.img))
      rclpy.spin_once(simple_pub_sub)
      await asyncio.sleep(1/ctx.feed_rate)
    simple_pub_sub.destroy_node()

  tg.asyncs.append(proc)
  with tg.enter(), rclpy_context():
    time.sleep(10)
    app.wait_or_interactive()

  #%%


def test_stream(ctx):

  context = zmq.Context()
  serverSocket = context.socket(zmq.PUB)
  port = 9872
  serverSocket.bind("tcp://*:" + str(port))
  t = 0.0
  while True:
    time.sleep(0.05)
    t += 0.05
    data = {
        "timestamp": t,
        "test_data":
            {
                "cos": np.cos(t),
                "sin": np.sin(t),
                "floor": np.floor(np.cos(t)),
                "ceil": np.ceil(np.cos(t))
            }
    }

    serverSocket.send_string(cmisc.json_dumps(data))


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
