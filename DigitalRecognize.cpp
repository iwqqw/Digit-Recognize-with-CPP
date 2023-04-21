#include <iostream>
#include "NeuralNet.h"
using namespace std;
network net;
int main()
{
	net.initConv();
	net.OpenPic("D:\\Project\\DigitalRecognize\\digit-recognizer\\train.csv",3);
	net.addConvolutionLayer(1, 6, 5, 1);
	net.addConvolutionLayer(2, 16, 5, 1);
	net.addConvolutionLayer(3, 120, 5, 1);
	net.convolution(1);
	net.pooling(1,0);
	net.convolution(2);
	net.pooling(2, 1);
	net.convolution(3);
	net.forward();
	net.printPic();
	//cout << "Hello CMake" << endl;
	return 0;
}