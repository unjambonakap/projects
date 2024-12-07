
sudo -s G_MESSAGES_DEBUG=all /usr/lib/geoclue                                                               │
python -u feeder.py  --actions=test | socat - UNIX-LISTEN:/var/run/benoit-daemon/geoclue-nmea.sock,mode=777,reuseaddr,fork

chown -R benoit:benoit /var/run/benoit-daemon/

/var/run/benoit-daemon/geoclue-nmea.sock

python -u feeder.py  --feed-speed 20 --verbosity=DEBUG --infile ~/Downloads/gr54-depuis-le-bourg-d-oisans.gpx --actions=test | socat - UNIX-LISTEN:/var/run/benoit-daemon/geoclue-nmea.sock,mode=777,reuseaddr,fork


/etc/geoclue/geoclue.conf
nmea-socket=/var/run/benoit-daemon/geoclue-nmea.sock

# Filesystem lockdown
ProtectSystem=strict
ProtectKernelTunables=true
ProtectControlGroups=true
ProtectHome=true
PrivateTmp=true

