#nohup python -u /home/bwtseng/Downloads/distiller/examples/classifier_compression/compress_classifier.py -a \
#			resnet20_cifar \
#			~/Downloads/data/cifar-10/batches-py/ -p=50 --lr=0.4 --epoch=10 \
#			--compress=/home/bwtseng/Downloads/distiller/examples/agp-pruning/resnet20_filters.schedule_agp.yaml \
#			--vs=0 --reset-optimizer --gpu=0  \
#			> distiller.out &
			#--resume-from=/home/bwtseng/Downloads/distiller/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar  \
# time python3 compress_classifier.py -a=mobilenet -p=50 --lr=0.001 ../../../data.imagenet/ -j=22 --resume-from=mobilenet_sgd_68.848.pth.tar --epochs=96 --compress=../agp-pruning//mobilenet.imagenet.schedule_agp.yaml  --reset-optimizer --vs=0


#python /home/bwtseng/Downloads/distiller/examples/classifier_compression/compress_classifier.py \
#		 -a=mobilenet \
#		 -p=50 --lr=0.001 -b 64 \
#		 /home/swai01/imagenet_datasets/raw-data/ \
#		 --resume-from=/home/bwtseng/Downloads/mobilenet_sgd_68.848.pth.tar \
#		 --epoch=96 \
#		 --compress=/home/bwtseng/Downloads/distiller/examples/agp-pruning/mobilenet.imagenet.schedule_agp.yaml \
#		 --reset-optimizer --vs=0 



#python  /home/bwtseng/Downloads/distiller/examples/classifier_compression/compress_classifier.py \
#		--arch resnet20_cifar  /home/bwtseng/Downloads/data/cifar-10/batches-py/ -p=50 --lr=0.4 --epochs=180 \
#		--compress=/home/bwtseng/Downloads/distiller/examples/agp-pruning/resnet20_filters.schedule_agp.yaml  \
#		--resume-from=/home/bwtseng/Downloads/distiller/examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar \
#		--vs=0 \
#		--reset-optimizer --gpu=0


python  /home/bwtseng/Downloads/distiller/examples/classifier_compression/compress_classifier.py \
		-a=alexnet --lr=0.005 -p=50 /home/swai01/imagenet_datasets/raw-data/ -b 32 --epochs=90 \
		--compress=/home/bwtseng/Downloads/distiller/examples/agp-pruning/alexnet.schedule_agp.yaml  \
		--pretrained -j 22