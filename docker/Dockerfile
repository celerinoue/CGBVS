FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER sotainoue

# lang
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Linux update & install custom module
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y sudo wget vim bzip2 curl git npm screen libfontconfig1 libxrender1 zsh

# Create user and set ownership and permissions as required*
ARG USERNAME=sotainoue
ARG GROUPNAME=sotainoue
ARG UID=1005
ARG GID=1005
ARG PASSWORD=sotainoue
RUN groupadd -g $GID $GROUPNAME \
    && useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME \
    && echo $USERNAME:$PASSWORD | chpasswd \
    && echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# env
SHELL ["/bin/bash", "-c"]
WORKDIR /home/$USERNAME/
RUN mkdir -p /home/$USERNAME/CGBVS

# =======
# package

# install conda
ENV CONDA_VERSION Miniconda3-py37_4.10.3-Linux-x86_64.sh
ENV CONDA_DIR /home/$USERNAME/miniconda
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# install
RUN INSTALL_PATH=/home/$USERNAME/miniconda \
    && wget --quiet https://repo.anaconda.com/miniconda/$CONDA_VERSION  -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -fbp $INSTALL_PATH
    # make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc
# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash


# other package
RUN conda install -y tensorflow-gpu==1.15.0 \
    && conda install -c conda-forge rdkit \
    && conda install -y scikit-learn==0.24.2 \
    && conda install -y joblib pandas keras matplotlib

RUN pip install mendeleev oddt seaborn biopython
# install kGCN
RUN pip install --upgrade git+https://github.com/celerinoue/kGCNv.git
RUN git clone https://github.com/celerinoue/kGCNv.git


# rm chche
#RUN apt-get autoremove -y \
#    && apt-get clean \
#    && rm -rf /usr/local/src/* \
#    && rm -rf /var/lib/apt/lists/*
#RUN rm -rf ~/miniconda.sh
#RUN rm -rf ~/.cache/pip
#RUN conda clean -i -t -y
