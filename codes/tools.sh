#! /usr/bin/env bash

#################################################################
#                    install pycharm                            #
#################################################################

sudo add-apt-repository ppa:mystic-mirage/pycharm
sudo apt update
sudo apt install pycharm


#################################################################
#                    install google                             #
#################################################################



#################################################################
#                  install virtualenv                           #
#################################################################
sudo apt-get install python3-pip
sudo pip3 install virtualenv virtualenvwrapper
mkdir -p $WORKON_HOME

#step1: open .bashrc 
#step2: add 1.export WORKON_HOME=$HMOE/.virtualenvs 
#           2.export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3 
#           3.source /usr/local/bin/virtualenvwrapper.sh

source ~/.bashrc


################################################################
#                   install youdao                             #
################################################################

# install youdao.pkg
# wget http://codown.youdao.com/cidian/linux/youdao-dict_1.1.0-0-deepin_amd64.deb
sudo apt-get -f install
sudo apt-get install python3-pyqt5
sudo apt-get install tesseract-ocr
# sudo dpkg -i youdao.pkg


################################################################
#               install docker on ubuntu16.04                  #
################################################################
sudo apt-get install -f
sudo apt install curl
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker

# use docker without sudo
sudo groupadd docker
sudo gpasswd -a ${USER} docker
sudo service docker restart

# set proxy for docker 
# step1:  touch  /etc/systemd/system/docker.service.d/http-proxy.conf
# add in http-proxy.conf  
#       1. [Service]
#       2. Environment="HTTP_PROXY=http:// your proxy :port"
#       3. Environment="HTTPS_PROXY=http:// your proxy :port"
# step2:  systemctl daemon-reload
# step3:  systemctl restart docker





################################################################
#                          open ssh                            #
################################################################

sudo apt-get install openssh-server
sudo service ssh start


################################################################
#              install and start mysql redis                   #
################################################################

# python connect mysql
# step1:  sudo apt-get install libmysqlclient-dev
# step2:  python2: pip install mysql-python  python3: pip install mysqlclient

# install mysql:  apt-get install mysql-server  start: service mysql start
# install redis:  apt-get install redis-server  start: sudo systemctl start redis




################################################################
#         install charles in ubuntu16.04                       #
################################################################
1.wget -q -O - https://www.charlesproxy.com/packages/apt/PublicKey | sudo apt-key add -
2.sudo sh -c 'echo deb https://www.charlesproxy.com/packages/apt/ charles-proxy main > /etc/apt/sources.list.d/charles.list'
3.sudo apt-get update
4.sudo apt-get install charles-proxy







