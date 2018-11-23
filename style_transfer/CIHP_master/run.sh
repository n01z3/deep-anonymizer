NV_GPU=0 nvidia-docker run -v `pwd`:/src -w "/src" -it floydhub/tensorflow:1.10.0-gpu.cuda9cudnn7-py2_aws.33 python run.py --img_list img_list1 --tta ''
