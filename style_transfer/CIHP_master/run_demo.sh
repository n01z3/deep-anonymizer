nvidia-docker run -it --rm -p 127.0.0.1:6164:6164 -v /home/dulyanov/projects/fiinsh/web/modules/combined:/seg_out -v `pwd`:/src fiinsh:cihp wrappa --disable-consul
