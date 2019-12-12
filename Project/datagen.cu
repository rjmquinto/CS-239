#include <fstream>
#include "util.h"

long long MAX_SIZE = (1ll<<25);
long long RELU_IN_SIZE = MAX_SIZE;
long long LSTM_IN_SIZE = MAX_SIZE;
long long LSTM_W_SIZE = (1ll<<12);
long long LSTM_B_SIZE = (1ll<<12);
long long LSTM_U_SIZE = (1ll<<12);

int main() {
	float *buffer;
	std::ofstream fout;
	buffer = (float*)malloc(MAX_SIZE*sizeof(float));


	randomize(buffer, RELU_IN_SIZE, -1, 1);
	fout.open("relu/in.dat", std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<char*>(&RELU_IN_SIZE), sizeof(RELU_IN_SIZE));
	fout.write(reinterpret_cast<char*>(buffer),RELU_IN_SIZE*sizeof(float));
	fout.close();
	

	randomize(buffer, LSTM_IN_SIZE, -1, 1);
	fout.open("lstm/in.dat", std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<char*>(&LSTM_IN_SIZE), sizeof(LSTM_IN_SIZE));
	fout.write(reinterpret_cast<char*>(buffer),LSTM_IN_SIZE*sizeof(float));
	fout.close();


	randomize(buffer, LSTM_W_SIZE, -1, 1);
	fout.open("lstm/w.dat", std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<char*>(&LSTM_W_SIZE), sizeof(LSTM_W_SIZE));
	fout.write(reinterpret_cast<char*>(buffer),LSTM_W_SIZE*sizeof(float));
	fout.close();

	randomize(buffer, LSTM_B_SIZE, -1, 1);
	fout.open("lstm/b.dat", std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<char*>(&LSTM_B_SIZE), sizeof(LSTM_B_SIZE));
	fout.write(reinterpret_cast<char*>(buffer),LSTM_B_SIZE*sizeof(float));
	fout.close();

	randomize(buffer, LSTM_U_SIZE, -1, 1);
	fout.open("lstm/u.dat", std::ios::out | std::ios::binary);
	fout.write(reinterpret_cast<char*>(&LSTM_U_SIZE), sizeof(LSTM_U_SIZE));
	fout.write(reinterpret_cast<char*>(buffer),LSTM_U_SIZE*sizeof(float));
	fout.close();

	free(buffer);
}
