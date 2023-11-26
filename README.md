# Neonatal Chest Sound Separation Application
The following is the repository for the application to perform chest sound separation. This application is based on our paper "Neonatal Chest Sound Separation using Deep Learning", which can be found [here](https://arxiv.org/abs/2310.17116). The application is create on [Streamlit](https://streamlit.io/).

## Application Link
The following [link](https://separate.yypoh.com/) will take you to the application website.

## Docker
The following application is uploaded to [DockerHub](https://hub.docker.com/r/yangyipoh/sep_app). The application can be deployed by executing the following command.

```
docker run -p 8501:8501 yangyipoh/sep_app
```

Then, you can access the application from a web browser by typing in

```
http://localhost:8051
```


## TODO
- [ ] Add signal quality assessment