docker build -t model_train .
#docker run --name mycontainer1   model_train
docker run -v "$()/app/models" model_train