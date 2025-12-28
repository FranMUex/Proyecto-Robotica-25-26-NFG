#! /usr/bin/bash

bash ~/robocomp/tools/rcnode/rcnode.sh &

cd /usr/local/bin

if ! test -f robocomp; then
  sudo ln -s /home/usuario/software/pip_env/bin/robocomp
  sudo ln -s /home/usuario/software/pip_env/bin/robocompdsl
fi

if ! killall -0 webots; then
  webots &
  disown
  sleep 3
fi

if ! killall -0 Webots2Robocomp; then
  cd ~/robocomp/components/webots-bridge/

  gnome-terminal --tab --execute bash -c "bin/Webots2Robocomp etc/config"
fi

if ! killall -0 Lidar3D; then
  cd ~/robocomp/components/robocomp-robolab/components/hardware/laser/lidar3D/

  gnome-terminal --tab --execute bash -c "bin/Lidar3D etc/config_helios_webots"
fi
