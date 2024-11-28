#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import ctypes
import time
import asyncio

os.environ['MAVLINK20'] = '1'

import asyncio
from mavsdk import System
from mavsdk.gimbal import GimbalMode, ControlMode
from pymavlink import mavutil
import io

import asyncio
from mavsdk import System
import mavsdk

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--dirname', default='/')
  parser.add_argument('--no-drone', action='store_true')
  parser.add_argument('--filename')
  parser.add_argument('--target')


def test(ctx):
  pass


async def list_ftp(ctx):
  print("directory list", await ctx.drone.ftp.list_directory(ctx.dirname))


async def download_ftp(ctx):
  d: mavsdk.System = ctx.drone

  assert ctx.filename

  assert ctx.filename is not None
  target = ctx.target
  if target is None:
    target = os.getcwd()

  async for prog in d.ftp.download(ctx.filename, target, False):
    print(prog)


async def upload_ftp(ctx):
  d: mavsdk.System = ctx.drone
  assert ctx.filename is not None
  assert ctx.target is not None
  async for prog in d.ftp.upload(ctx.filename, ctx.target):
    print(prog)
  print('done')


async def calibrate_esc(ctx):
  d: mavsdk.System = ctx.drone
  from pymavlink.dialects.v20 import ardupilotmega as ml
  conn = mavutil.mavlink_connection(
      'tcp:raspberrypi.local:12346', source_system=1, source_component=3
  )

  conn.mav.battery_status_send(
      id=124,
      battery_function=ml.MAV_BATTERY_FUNCTION_UNKNOWN,
      type=ml.MAV_BATTERY_TYPE_LION,
      temperature=45_00,
      voltages=[2**16-1] * 10,
      mode=ml.MAV_BATTERY_MODE_UNKNOWN,
      current_battery=1_000,
      current_consumed=100,
      energy_consumed=-1,
      battery_remaining=-1
  )
  await asyncio.sleep(0.1)

  async for battery in d.telemetry.battery():
    print(f"Battery: {battery}")
    break
  async for x in d.calibration.calibrate_esc(False):
    print('GOOT', x)
    if x.status_text == 'Connect battery now':
      input('Connect battery now')
      conn.mav.battery_status_send(
          id=124,
          battery_function=ml.MAV_BATTERY_FUNCTION_UNKNOWN,
          type=ml.MAV_BATTERY_TYPE_LION,
          temperature=45_00,
          voltages=[10_000] * 10,
          mode=ml.MAV_BATTERY_MODE_UNKNOWN,
          current_battery=1_000,
          current_consumed=100,
          energy_consumed=-1,
          battery_remaining=-1
      )

  #x = res.pack(m)
  #m.decode(bytearray(x))
  #conn.write(

  return
  d: mavsdk.System = ctx.drone
  await d.param.set_param_int('BAT_OVRD_CONN', 1)
  print(await d.param.get_param_int('MAV_1_CONFIG'))


async def print_info(ctx):
  d: mavsdk.System = ctx.drone
  return

  async for battery in d.telemetry.battery():
    print(f"Battery: {battery}")
  return

  u = d.shell.receive()
  print(await d.shell.send('mavlink status'))
  async for x in u:
    print(x)

  return
  async for position in ctx.drone.telemetry.position():
    print(position)
  print('DONe get position')
  sys.exit(0)

  async for gps_info in ctx.drone.telemetry.gps_info():
    print(f"GPS info: {gps_info}")


async def get_imu_data():
  # Connect to the drone
  drone = System()
  await drone.connect(system_address="serial:///dev/ttyUSB0:57600")

  # Wait for the drone to connect
  print("Waiting for drone to connect...")
  async for state in drone.core.connection_state():
    if state.is_connected:
      print("Drone is connected!")
      break

  telemetry = drone.telemetry

  # Set the rate at which IMU data is updated (in Hz)
  await telemetry.set_rate_imu(200.0)

  # Fetch and print IMU data
  print("Fetching IMU data...")
  async for imu in telemetry.imu():
    # Print data in HIGHRES_IMU format
    print(f"HIGHRES_IMU (105)")
    print(f"Time (us): {imu.timestamp_us}")
    print(f"X Acceleration (m/s^2): {imu.acceleration_frd.forward_m_s2}")
    print(f"Y Acceleration (m/s^2): {imu.acceleration_frd.right_m_s2}")
    print(f"Z Acceleration (m/s^2): {imu.acceleration_frd.down_m_s2}")
    print(f"X Gyro (rad/s): {imu.angular_velocity_frd.forward_rad_s}")
    print(f"Y Gyro (rad/s): {imu.angular_velocity_frd.right_rad_s}")
    print(f"Z Gyro (rad/s): {imu.angular_velocity_frd.down_rad_s}")
    print(f"X Mag (gauss): {imu.magnetic_field_frd.forward_gauss}")
    print(f"Y Mag (gauss): {imu.magnetic_field_frd.right_gauss}")
    print(f"Z Mag (gauss): {imu.magnetic_field_frd.down_gauss}")
    print(f"Temperature (°C): {imu.temperature_degc}")
    print("-----------------------------------------")


async def opa_ah_init(ctx):
  if ctx.no_drone: return

  drone = System()
  await drone.connect('tcpout://192.168.131.223:12345')
  #await drone.core.set_mavlink_timeout(0.5)
  #ActionHandler.g_handler.stack.callback(drone._stop_mavsdk_server)
  print("Waiting for drone to connect...")
  async for state in drone.core.connection_state():
    if state.is_connected:
      print(f"-- Connected to drone!")
      break

  ctx.drone = drone


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)
  ctx.clear()


app()
