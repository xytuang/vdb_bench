#!/bin/bash

INTERVAL=1
NET_IFACE="eno1d1"
DISK_DEV="dm-0"  # /dev/mapper/emulab-bs1

echo "Monitoring network ($NET_IFACE) and disk ($DISK_DEV) I/O every ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""
printf "%-25s %12s %12s %12s %12s\n" "Time" "Net RX MB/s" "Net TX MB/s" "Disk R MB/s" "Disk W MB/s"
echo "------------------------------------------------------------------------------------"

while true; do
    # Network
    rx1=$(cat /proc/net/dev | awk "/^\s*${NET_IFACE}:/{print \$2}")
    tx1=$(cat /proc/net/dev | awk "/^\s*${NET_IFACE}:/{print \$10}")

    # Disk: fields in /proc/diskstats are:
    # col 6 = sectors read, col 10 = sectors written (512 bytes each)
    read1=$(awk "\$3==\"${DISK_DEV}\"{print \$6}" /proc/diskstats)
    write1=$(awk "\$3==\"${DISK_DEV}\"{print \$10}" /proc/diskstats)

    sleep $INTERVAL

    rx2=$(cat /proc/net/dev | awk "/^\s*${NET_IFACE}:/{print \$2}")
    tx2=$(cat /proc/net/dev | awk "/^\s*${NET_IFACE}:/{print \$10}")
    read2=$(awk "\$3==\"${DISK_DEV}\"{print \$6}" /proc/diskstats)
    write2=$(awk "\$3==\"${DISK_DEV}\"{print \$10}" /proc/diskstats)

    net_rx=$(( (rx2 - rx1) / 1024 / 1024 ))
    net_tx=$(( (tx2 - tx1) / 1024 / 1024 ))
    disk_r=$(( (read2 - read1) * 512 / 1024 / 1024 ))
    disk_w=$(( (write2 - write1) * 512 / 1024 / 1024 ))

    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    printf "%-25s %12d %12d %12d %12d\n" "$timestamp" "$net_rx" "$net_tx" "$disk_r" "$disk_w"
done
