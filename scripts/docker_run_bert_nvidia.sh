PWD=`pwd`
DOCKER_DIR=/`basename $PWD`
docker run -it --network=host --runtime=nvidia -v=$PWD:$DOCKER_DIR -w $DOCKER_DIR bert_nvidia
