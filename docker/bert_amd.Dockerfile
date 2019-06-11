FROM rocm/tensorflow:rocm2.9-tf1.15-dev
RUN apt-get update && apt-get install -y \
    unzip \