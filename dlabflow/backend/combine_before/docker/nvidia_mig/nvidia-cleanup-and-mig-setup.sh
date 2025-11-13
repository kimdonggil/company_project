#!/bin/bash

# Set password
PASSWORD="@)@$rkddnjs!!"

# Kill processes using /dev/nvidia* if they are Xorg
for pid in $(echo "$PASSWORD" | sudo -S fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u); do
    if ps -p $pid -o comm= | grep -q '^Xorg$'; then
        echo "Killing Xorg process: PID $pid"
        echo "$PASSWORD" | sudo -S kill -9 $pid
    fi
done

# Enable MIG mode on GPU 1
echo "Enabling MIG on GPU 1..."
echo "$PASSWORD" | sudo -S nvidia-smi -i 1 -mig 1

# Create MIG instance
echo "Creating MIG instance with configuration 9,3g.40gb..."
echo "$PASSWORD" | sudo -S nvidia-smi mig -cgi 9,3g.40gb -C

