#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <ctime>
class network {
private:
	struct kernel {
		int kersize;
		double kern[32][32];
		double bias=1;
		double output[32][32] = {0};
		int outputsize;
		int stride;
		bool is_pooling = false;
		double err[32][32] = {0};
	};
	struct poolcore{
		int kersize;
		int TYPE;
		double bias=0;

		int stride=2;
	};
	std::vector <kernel> convk[10];
	poolcore poolc[2];
	double fullConnect1[121][85], fullConnect2[85][11];
	double neural[85],output[11];
public:
	void OpenPic(std::string PATH,int x);
	void printPic();
	void initConv();
	void addConvolutionLayer(int layer,int num,int kersize,int stride);
	void convolution(int layer);
	void pooling(int layerout, int layercoor);
	void forward();
	void backward();
private:
	int label;
	double weight_base_calc(int layer, int num, int kersize);
	void fanzhuan(kernel *kk);
	void randomkernel(kernel* kk,int layer, int kersize, int num);
	std::ifstream csv_data;
	double pictrue[32][32];
	//std::vector <std::vector <conv> > convlayer;
	double relu(double x);
	double crossEntropy();
	double dsrelu(double x);
	double sigmoid(double x);
	double dsigmoid(double x);
	
};