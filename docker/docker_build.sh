# build image
#docker build -t cgbvs_gpu ./docker

# run
docker run -it -d \
--runtime=nvidia \
--gpus all \
--name cgbvs_env \
-v /data_st02/drug/inoue/CGBVS:/home/sotainoue/CGBVS \
-v /etc/group:/etc/group:ro \
-v /etc/passwd:/etc/passwd:ro \
-u $(id -u $USER):$(id -g $USER) \
cgbvs_gpu bash


# run as root
'''
#docker run -it -d \
#--runtime=nvidia \
#--gpus all \
#--name cgbvs_env_root \
#-v /data_st02/drug/inoue/CGBVS:/home/sotainoue \
#cgbvs_gpu
'''


# activate
docker start cgbvs_env
docker attach cgbvs_env
#docker exec -it cgbvs_env cgbvs_gpu


# view
#docker images
