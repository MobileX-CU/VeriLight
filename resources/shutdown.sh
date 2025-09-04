# script to properly shut down the Raspberry Pi before unplugging it and powering down DLP230NP EVM

raspi-gpio set 25 op
raspi-gpio set 25 dl
sudo shutdown now
