AT BOARD_ID

frequency hopping done in the firmware - fhop_receive_channel
most likely radio_443x (ifndef CPU_SI1060)
https://community.silabs.com/s/article/ezradiopro-auto-frequency-hopping-4-wds-configuration-state-machine-and-time?language=en_US

idea: listen in firmware for mavlink command in order to control the radio

opendronemap



configure on FC
nsh> 

echo DEVICE=eth0 > /fs/microsd/net.cfg
echo BOOTPROTO=static >> /fs/microsd/net.cfg
echo IPADDR=192.168.2.3 >> /fs/microsd/net.cfg
echo NETMASK=255.255.255.0 >> /fs/microsd/net.cfg
echo ROUTER=192.168.2.1 >> /fs/microsd/net.cfg
echo DNS=10.0.0.1 >> /fs/microsd/net.cfg


nsh> param show
// to see MAV_1_CONFIG


RPI:
 socat UDP-LISTEN:12345,fork,reuseaddr /dev/serial0,b921600,raw
socat  UDP-LISTEN:12345,fork,reuseaddr UDP:192.168.2.3:14550,sp=14550

UDP:(net.cfg>IPADDR):MAV_2_UDP_PRT,sp=MAV_2_REMOTE_PRT

socat TCP-LISTEN:12345,fork,reuseaddr /dev/serial0,b921600,raw



laptop:
./Tools/mavlink_shell.py udpout:raspberrypi.local:12345
03/10
at night, updating px4 firmware v1.15 + setting frame (x500 v2) -> can't connect to mavlink on CM4 (udp + serial)
mind breaking

before sleeping, thought about parameters reinit - indeed MAV_1_CONFIG was fucked up

04/10

Would be great to be able to store parameters on sdcard - have not found a way
used TELEM1 (radios) to connect to nsh
 
also had issues with netman - BOOTPROTO=fallback -> was trying dhcp, then falling back to static. Introducting a lot of latency, waiting for dhcp to fail. With =static, much faster to connect to the mavlink udp instance



Then udp was not working once socat was killed and relaunched: (MAV_2_REMOTE_PRT), socat  UDP-LISTEN:12345,fork,reuseaddr UDP:192.168.2.3:14550,sp=14550
does the trick. Need to listen on port 14550 as well (without sp, that fails)



Calibration ESC-> not working QGC. Test it using mavsdk, mavsdk-python
it does not work out of the box - calibration only exists for gyro, accel +1

fucking mavsdk server not working  with udp://raspberrypi.local:12345 - investigating it only works giving an IP adress -> rabbit hole to fix it
```
# /home/benoit/.virtualenvs/env3.12/lib/python3.12/site-packages/mavsdk/bin/mavsdk_server -p 50051 --sysid 245 --compid 190 udp://raspberrypi.local:12345
drone = System(mavsdk_server_address='localhost')
print('laa')
await drone.connect()
```


-> means compiling mavsdk, mavsdk-python, mavlink, mav-proto
because used version of protobuf is 3, should update it (more dfs)


TODO:
can publish mavlink messge for battery_status. But it sets connected=true always
esc_calibration.cpp -> modify battery disconnect check so that it's a state we can make believe by publishing battery_status inf
Can do: cell_count > 0, voltage = 0


05/10
climbing
do_esc_calibration refactoring
How is the proress reported?
It's basically an RPC with status text sent, intended as being interactive


mavlink has COMMAND_LONG with COMMAND_ACK (for long lived rpc, progress)
these upon receptions in px4 (MavlinkReceiver::handle_message_command_long(mavlink_message_t *msg)) are transformed as vehicle_command_s (exactly same? - internal type) (vehicle_command_ack_s exists as well).
commander listen to the publisher of vehicle_command_s
on VEHICLE_CMD_PREFLIGHT_CALIBRATION, it performs esc calibration. Returns ACCEPTED directly (denied in some cases - bad)
Then really starts the RPC (which is not really finished when ACCEPTED is received. QCG does not treat it as regular command (though it returns directly :o))
Progress reports are sent as STATUS_TEXT (through logging). They are listened to in QCG/mavsdk, "[cal] done/failed/progress
Really bad if start multiple calibrations simultaneously


06/10
mavsdk_server started by mavsdk-python directly still does not show anything
reason: mavsdk_server buffers shit -> needs to run it unbuffered (stdbuf) . Have not found how to do it portably - how does stdbuf work under the hood? setvbuf in python could work?




python uav_py/siyi_test.py --actions=test_controller
ffplay rtsp://192.168.1.25:8554/main.264




sik radio
adding interceptor

LD obj/hm_trp/radio~hm_trp/radio~hm_trp.ihx

?ASlink-Error-Insufficient EXTERNAL RAM memory.

Internal RAM layout:
      0 1 2 3 4 5 6 7 8 9 A B C D E F
0x00:|0|0|0|0|0|0|0|0|a|a|a|a|a|b|b|c|
0x10:|c|d|d|d|d|d|d|d|d|e|e|e|h|h|h|h|
0x20:|B|B|B|B|B|T|f|f|f|f|f|f|f|f|f|f|
0x30:|f|f|f|f|f|f|f|f|f|f|f|g|g|g|g|g|
0x40:|g|g|g|g|g|g|g|g|g|g|i|i|i|i|i|i|
0x50:|i|i|i|i|j|j|j|k|k|k|k|k|k|k|k|k|
0x60:|k|k|k|k|k|k|k|k|k|k|k|k|k|k|k|k|
0x70:|k|k|l|l|l|m|m|m|m|Q|Q|Q|Q|Q|I|I|
0x80:|I|I|I|I|I|I|I|I|I|I|S|S|S|S|S|S|
0x90:|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|
0xa0:|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|
0xb0:|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|S|
0xc0:|S|S|S|S|S|S|S|S|S|S| | | | | | |
0xd0:| | | | | | | | | | | | | | | | |
0xe0:| | | | | | | | | | | | | | | | |
0xf0:| | | | | | | | | | | | | | | | |
0-3:Reg Banks, T:Bit regs, a-z:Data, B:Bits, Q:Overlay, I:iData, S:Stack, A:Absolute

Stack starts at: 0x8a (sp set to 0x89) with 64 bytes available.
The largest spare internal RAM space starts at 0xca with 54 bytes available.

Other memory:
   Name             Start    End      Size     Max     
   ---------------- -------- -------- -------- --------
   PAGED EXT. RAM   0x0001   0x00c0     192      256   
   EXTERNAL RAM     0x00c1   0x1005    3909     4096   
   ROM/EPROM/FLASH  0x0000   0xab8d   42896    62464   
*** ERROR: Insufficient EXTERNAL RAM memory.



erf.
solution: in serial.c, reduce RX_BUF_MAX



need to patch sip4 for python3.12
-        frame = frame->f_back;
+        frame = PyFrame_GetBack(frame);

then ros jazzy compiles fine



on RPI, to talk to siyi on 192.168.1.25
sudo ip addr add 192.168.1.1 dev eth0


rsync --progress -Lavz /home/benoit/programmation/hack/chdrft// /home/benoit/sshfs/chdrft/
(-L for symlinks)



mediamtx.yml
paths:
  "~^proxy_(.+)$":
    # If path name is a regular expression, $G1, G2, etc will be replaced
    # with regular expression groups.
    source: rtsp://192.168.1.25:8554/$G1
    sourceOnDemand: yes



2024/11/23 12:35:31 WAR [path proxy_main.264] [RTSP source] RTP packet is too big to be read with UDP

sudo ip link set wlan0 mtu 1500 -> to no avail

iftrap -> monitor packet size
sudo setcap cap_net_raw=eip ~/programmation/binary/cap_python
cap_python uav_py/test_scapy.py  --actions=test





#0  0x00007ffff6ea53f4 in ?? () from /usr/lib/libc.so.6
#1  0x00007ffff6e4c120 in raise () from /usr/lib/libc.so.6
#2  0x00007ffff6e334c3 in abort () from /usr/lib/libc.so.6
#3  0x00005555555bd573 in uORB::DeviceNode::open (this=0x7fffdc000b70, filp=0x555555a2c3a0 <filemap>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/uORBDeviceNode.cpp:84
#4  0x0000555555867092 in px4_open (path=<optimized out>, flags=0x2) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/cdev/posix/cdev_platform.cpp:214
#5  0x00005555557d10d5 in uORB::Manager::node_open (this=0x555555a8d9e0, meta=meta@entry=0x555555a19da0 <__orb_dataman_request>, advertiser=advertiser@entry=0x1, instance=0x0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/uORBManager.cpp:525
#6  0x00005555557d1138 in uORB::Manager::orb_advertise_multi (this=<optimized out>, meta=0x555555a19da0 <__orb_dataman_request>, data=0x0, instance=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/uORBManager.cpp:295
#7  0x000055555585f693 in uORB::Publication<dataman_request_s>::advertise (this=0x7fff80005778) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/Publication.hpp:98
#8  DatamanClient::DatamanClient (this=0x7fff80005740) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/dataman_client/DatamanClient.cpp:44
#9  0x000055555585fe7b in DatamanCache::DatamanCache (this=this@entry=0x7fff80005728, cache_miss_perf_counter_name=cache_miss_perf_counter_name@entry=0x5555558995f8 "geofence_dm_cache_miss", num_items=num_items@entry=0x0) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/dataman_client/DatamanClient.cpp:444
#10 0x000055555572f8cb in Geofence::Geofence (this=this@entry=0x7fff800056e8, navigator=navigator@entry=0x7fff80004f90) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/navigator/geofence.cpp:79
#11 0x00005555557193d6 in Navigator::Navigator (this=0x7fff80004f90) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/navigator/navigator_main.cpp:74
#12 0x000055555571a622 in Navigator::instantiate (argc=<optimized out>, argv=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/navigator/navigator_main.cpp:1094
#13 ModuleBase<Navigator>::run_trampoline (argc=<optimized out>, argv=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/include/px4_platform_common/module.h:176
#14 0x000055555586f7b7 in entry_adapter (ptr=0x555555b20650) at /home/benoit/repos/drone/PX4-Autopilot/platforms/posix/src/px4/common/tasks.cpp:98
#15 0x00007ffff6ea339d in ?? () from /usr/lib/libc.so.6
#16 0x00007ffff6f2849c in ?? () from /usr/lib/libc.so.6


#0  0x00007ffff6ea53f4 in ?? () from /usr/lib/libc.so.6
#1  0x00007ffff6e4c120 in raise () from /usr/lib/libc.so.6
#2  0x00007ffff6e334c3 in abort () from /usr/lib/libc.so.6
#3  0x00005555555bd59f in uORB::DeviceNode::close (this=0x7fffdc000b70, filp=0x555555a2b3a0 <filemap>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/uORBDeviceNode.cpp:127
#4  0x0000555555866fb3 in px4_close (fd=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/cdev/posix/cdev_platform.cpp:255
#5  0x0000555555790fc9 in SendTopicsSubs::reset (this=0x7fff7c0025b0) at /home/benoit/repos/drone/PX4-Autopilot/build/px4_sitl_default/src/modules/uxrce_dds_client/dds_topics.h:313
#6  UxrceddsClient::delete_session (this=this@entry=0x7fff7c000b70, session=session@entry=0x7fffdbdff780) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/uxrce_dds_client/uxrce_dds_client.cpp:384
#7  0x0000555555792c23 in UxrceddsClient::run (this=0x7fff7c000b70) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/uxrce_dds_client/uxrce_dds_client.cpp:527
#8  0x00005555557933d8 in ModuleBase<UxrceddsClient>::run_trampoline (argc=<optimized out>, argv=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/include/px4_platform_common/module.h:180
#9  0x000055555586f4f7 in entry_adapter (ptr=0x7fff8c00ae00) at /home/benoit/repos/drone/PX4-Autopilot/platforms/posix/src/px4/common/tasks.cpp:98
#10 0x00007ffff6ea339d in ?? () from /usr/lib/libc.so.6
#11 0x00007ffff6f2849c in ?? () from /usr/lib/libc.so.6



#0  0x000056b995331034 in uORB::Subscription::valid (this=0x0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/Subscription.hpp:118
#1  uORB::Subscription::advertised (this=0x0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/Subscription.hpp:121
#2  uORB::SubscriptionInterval::advertised (this=0x0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/SubscriptionInterval.hpp:91
#3  uORB::SubscriptionInterval::updated (this=0x0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/SubscriptionInterval.cpp:41
#4  0x000056b995333acd in uORB::DeviceNode::poll_state (this=<optimized out>, filp=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/platforms/common/uORB/uORBDeviceNode.cpp:335
#5  0x000056b9953ca611 in cdev::CDev::poll (this=this@entry=0x7bd594006600, filep=0x56b99558f380 <filemap>, fds=fds@entry=0x7bd5f33ff820, setup=setup@entry=0x1) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/cdev/CDev.cpp:288
#6  0x000056b9953cb06a in px4_poll (fds=0x7bd5f33ff820, nfds=<optimized out>, timeout=0x3e8) at /home/benoit/repos/drone/PX4-Autopilot/src/lib/cdev/posix/cdev_platform.cpp:375
#7  0x000056b99517c21f in task_main (argc=<optimized out>, argv=<optimized out>) at /home/benoit/repos/drone/PX4-Autopilot/src/modules/dataman/dataman.cpp:706
#8  0x000056b9953d3277 in entry_adapter (ptr=0x7bd5e80030f0) at /home/benoit/repos/drone/PX4-Autopilot/platforms/posix/src/px4/common/tasks.cpp:98
#9  0x00007bd5fbca339d in ?? () from /usr/lib/libc.so.6
#10 0x00007bd5fbd2849c in ?? () from /usr/lib/libc.so.6



ros2 topic list --no-daemon
MicroXRCEAgent  udp4 -p 8888  -v 4 -r agent_profile_dds.xml
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/ros_profile_dds.xml
export ROS_DISCOVERY_SERVER="127.0.0.1:11811"
fastdds discovery --server-id 0 -p 11811
param set UXRCE_DDS_PTCFG 2





from .config -> config.h
./platforms/nuttx/NuttX/nuttx/include/nuttx/config.h

platforms/nuttx/NuttX/nuttx/tools/mkconfig.c|92 col 3| generate_definitions(stream);
./build/px4_fmu-v6x_default/NuttX/nuttx/.config



todo:
on shutdown (before  px4_shutdown finishes properly), disable lockstep shit

setpgid -> system() spawns in same thrae group -> SIGINT gets propagated even though we trap it in the parent process



valgrind --leak-check=full --show-leak-kinds=all --gen-suppressions=all --log-file=valgrind.out ./build/px4_sitl_default/bin/px4

git update-index --assume-unchanged .editorconfig

