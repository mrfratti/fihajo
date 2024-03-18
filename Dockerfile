FROM jenkins/jenkins:lts

USER root

RUN jenkins-plugin-cli --plugins "json-path-api"

RUN jenkins-plugin-cli --plugins "blueocean docker-workflow"


# Why remove the entire apt package lists? 
RUN apt-get update && \
  apt-get install -y python3 python3-pip python3-venv python-is-python3 
#&& \
#  ln -s /usr/bin/python3 /usr/bin/python  
# && \ 
#apt-get clean && \
#rm -rf /var/lib/apt/lists/* 

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN pip install --no-cache-dir \
  tensorflow==2.15.0 \
  uncertainty-wizard==0.4.0 \
  h5py==3.10.0 \
  jupyter==1.0.0 \
  keras==2.15.0 \
  matplotlib==3.8.2 \
  numpy==1.26.3 \
  pandas==2.2.0 \
  scikit-learn==1.4.0 \
  seaborn==0.13.2 \
  cleverhans==4.0.0 \
  DateTime==5.4 \
  argparse==1.4.0


USER jenkins


WORKDIR "/var/jenkins_home/workspace"
# copy from local machine to docker image
COPY src/ src/ 
