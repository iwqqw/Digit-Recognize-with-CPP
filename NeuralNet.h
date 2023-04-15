#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
class network {
private:
	struct kernel {
		int kersize;
		double kern[32][32];
		double bias=0;
		double output[32][32];
		int outputsize;
		int stride;
	};
	struct poolcore{
		int kersize;
		int TYPE;
		double bias=0;

		int stride=2;
	};
	std::vector <kernel> convk[10];
	poolcore poolc[2];
public:
	void OpenPic(std::string PATH,int x);
	void printPicture();
	void initConv();
	void convolution(int layer);
	void pooling(int layerout, int layercoor);
private:
	int label;
	void fanzhuan(kernel *kk);
	std::ifstream csv_data;
	double pictrue[32][32];
	//std::vector <std::vector <conv> > convlayer;
	double relu(double x);
	double desrelu(double x);
};