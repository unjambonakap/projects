#!/usr/bin/env python3

import asyncio
from mavsdk import System
import logging

logging.getLogger().setLevel(logging.DEBUG)


async def run():

  # needs to be when mavsdk_server is not run separetly
  #drone = System(mavsdk_server_address='udpout://192.168.205.223:12345')
  if 0:
    drone = System(mavsdk_server_address='localhost')
    await drone.connect()
  else:
    drone = System()
    await drone.connect('tcpout://192.168.131.223:12345')
    return
  # Wait for the drone to connect
  print("Waiting for drone to connect...")
  async for state in drone.core.connection_state():
    if state.is_connected:
      print("Drone is connected!")
      break

  #async for position in drone.telemetry.position():
  #    print(position)
  print("directory list", await drone.ftp.list_directory('/'))
  return
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
  return

  print("-- Starting gyroscope calibration")
  async for progress_data in drone.calibration.calibrate_gyro():
    print(progress_data)
  print("-- Gyroscope calibration finished")

  #print("-- Starting accelerometer calibration")
  #async for progress_data in drone.calibration.calibrate_accelerometer():
  #    print(progress_data)
  #print("-- Accelerometer calibration finished")

  #print("-- Starting magnetometer calibration")
  #async for progress_data in drone.calibration.calibrate_magnetometer():
  #    print(progress_data)
  #print("-- Magnetometer calibration finished")

  #print("-- Starting board level horizon calibration")
  #async for progress_data in drone.calibration.calibrate_level_horizon():
  #    print(progress_data)
  #print("-- Board level calibration finished")


if __name__ == "__main__":
  # Run the asyncio loop
  asyncio.run(run())
