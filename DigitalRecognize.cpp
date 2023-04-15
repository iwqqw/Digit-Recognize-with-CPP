#include <iostream>
#include "NeuralNet.h"
using namespace std;
network net;
int main()
{
	net.initConv();
	net.OpenPic("D:\\Project\\DigitalRecognize\\digit-recognizer\\train.csv",3);
	
	net.convolution(1);
	net.pooling(1,0);
	//cout << "Hello CMake" << endl;
	
}