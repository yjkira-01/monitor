@echo off
title Cloud Training Monitor
ssh -t -p 22 -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=%~dp0..\artifacts\cloud_known_hosts" root@your-cloud-host "/root/miniconda3/bin/python /root/autodl-fs/yanglao-monitor/scripts/monitor_training.py --interval 1"
echo.
echo Monitor closed. The background training is independent of this window.
pause
