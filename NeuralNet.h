#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
class network {
private:
	struct Layer {

	};
public:
	void OpenPic(std::string PATH,int x);
	void printPicture();
	void convolution(int step,int kernelsize,conv kern);
private:
	int label;
	std::ifstream csv_data;
	double pictrue[32][32];
	//std::vector <std::vector <conv> > convlayer;
	double relu(double x);
	double desrelu(double x);
};