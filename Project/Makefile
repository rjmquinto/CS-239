MAIN_OUT=main
DATAGEN_OUT=datagen
MAIN_RELU_OUT=main_relu
MAIN_LSTM_OUT=main_lstm

main: main.o util.o relu.o lstm.o
	nvcc main.o util.o relu.o lstm.o -o ${MAIN_OUT}

main_relu: main_relu.o util.o relu.o
	nvcc main_relu.o util.o relu.o -o ${MAIN_RELU_OUT}

main_lstm: main_lstm.o util.o lstm.o
	nvcc main_lstm.o util.o lstm.o -o ${MAIN_LSTM_OUT}

data: datagen
	mkdir -p lstm relu
	./${DATAGEN_OUT}

datagen: datagen.o util.o
	nvcc datagen.o util.o -o ${DATAGEN_OUT}
	chmod +x ${DATAGEN_OUT}

main.o: main.cu
	nvcc -c main.cu
util.o: util.cu
	nvcc -c util.cu
relu.o: relu.cu
	nvcc -c relu.cu
lstm.o: lstm.cu
	nvcc -c lstm.cu
datagen.o: datagen.cu
	nvcc -c datagen.cu
main_relu.o: main_relu.cu
	nvcc -c main_relu.cu
main_lstm.o: main_lstm.cu
	nvcc -c main_lstm.cu

clean all:
	rm -r ${MAIN_OUT} ${DATAGEN_OUT} *.o lstm relu

# test main_relu: main_relu
# 	./${MAIN_RELU_OUT}

# test main_lstm: main_lstm
# 	./${MAIN_LSTM_OUT}
