#include "NeuralNet.h"
void network::OpenPic(std::string PATH,int x)
{
	std::string line;
	
	csv_data.open(PATH, std::ios::in);
	std::istringstream iss;
	//iss.str(line);
	std::string num;
	for (int i = 0; i < x; i++)
	{
		std::getline(csv_data, line);
	}
	std::getline(csv_data, line);
	iss.clear();
	iss.str(line);
	int cont = 0;
	while (std::getline(iss, num, ',')) //���ַ�����sin�е��ַ�����field�ַ����У��Զ���Ϊ�ָ���
	{
		pictrue[cont / 28][cont % 28] = atol(num.c_str());
		// std::cout << atol(word.c_str());
		++cont;
	}
	//std::cout << line << std::endl;
}
void network::convolution(int step, int kernelsize,conv kern)
{

}