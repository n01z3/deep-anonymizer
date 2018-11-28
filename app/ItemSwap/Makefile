IMAGE=digitman/pytorch_tf_gpu

DOCKER_BUILD=docker build -t ${IMAGE} -f Dockerfile.gpu .
DOCKER_RUN=docker run --rm -it -v ${CURDIR}:/app ${IMAGE}

docker-build:
	${DOCKER_BUILD}

docker-push:
	docker push ${IMAGE}

run-bash:
	${DOCKER_RUN} /bin/bash

run-jupyter:
	docker run --rm -it -v ${CURDIR}:/app -w /app -p 8892:8892 ${IMAGE} jupyter notebook \
		--ip=0.0.0.0 --port=8892 --no-browser --allow-root \
		--NotebookApp.token='' --NotebookApp.password=''

load-prepare-face-data:
	export urls_path=urls_faces_sobchak.yaml
	export images_path=align_celeba
	export data_folder_path=data_all2sobchak
	${DOCKER_RUN} python load_prepare_face_data.py

clear-face-data:
	rm data_faces/videos/* -rf && rm data_faces/binary_masks -rf && rm data_faces/facesA -rf \
		&& rm data_faces/facesB -rf && rm data_faces/null.mp4 \
		&& rm weights_faces/gan_models -rf

convert-face-video:
	${DOCKER_RUN} python convert_face_video.py elon.mp4 gen_b.mp4 \
		--start_time 20 --end_time 25

convert-face-image:
	${DOCKER_RUN} python convert_face_image.py me.jpg gen_me.jpg

convert-background-image:
	${DOCKER_RUN} python convert_background.py --dataroot data_background \
	--name rick_morty --model cycle_gan --checkpoints_dir weights_background \
	--results_dir results --resize_or_crop none --loadSize 720

save-weights:
	zip -r ../gan_weights/weights_faces.zip weights_faces
	zip -r ../gan_weights/weights_background.zip weights_background

load-weights:
	wget https://bitbucket.org/EvgenyKashin/gan_weights/raw/53f9086c0104bd717d594cb5c8a943a61c579a70/weights_background.zip
	wget https://bitbucket.org/EvgenyKashin/gan_weights/raw/53f9086c0104bd717d594cb5c8a943a61c579a70/weights_faces.zip
	unzip weights_background.zip
	unzip weights_faces.zip
	rm weights_background.zip
	rm weights_faces.zip