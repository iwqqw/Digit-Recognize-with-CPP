#include "NeuralNet.h"
double network::relu(double x)
{
	return std::max(0.0, x);
}
double network::desrelu(double x)
{
	if (x > 0)
		return 1;
	else
		return 0;
}
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
	convk[0][0].outputsize = 32;
	while (std::getline(iss, num, ',')) //将字符串流sin中的字符读到field字符串中，以逗号为分隔符
	{
		convk[0][0].output[cont / 28][cont % 28] = atol(num.c_str());
		// std::cout << atol(word.c_str());
		++cont;
	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			std::cout << convk[0][0].output[i][j] << " ";
		}
		std::cout << std::endl;
	}
	//std::cout << line << std::endl;
}
void network::fanzhuan(kernel *kk)
{
	// 把m赋值给k，防止对后续操作产生影响
	int k = kk->kersize;
	for (int i = 0; i < kk->kersize; i++)
	{
		for (int j = 0; j < k; j++)
			// 交换这两个位置
			std::swap(kk->kern[i][j], kk->kern[kk->kersize - i - 1][kk->kersize - j - 1]);
		// 下次比较的次数少一位
		k--;
		// 此时不需要进行替换
		if (k == 1)
			break;
	}
	
}
void network::initConv()
{
	poolc[0].kersize = 2;
	poolc[0].TYPE = 1;
	poolc[0].stride = 2;
	poolc[1].kersize = 2;
	poolc[1].TYPE = 1;
	poolc[1].stride = 2;

	kernel kk;
	memset(kk.output, 0, sizeof(kk.output));
	kk.stride = 1;
	kk.outputsize = 32;
	//cout<<1;
	convk[0].push_back(kk);
	//fanzhuan(&convk[0][0]);
	kk.kersize = 3;
	kk.stride = 1;
	kk.kern[0][0] = 1;
	kk.kern[0][1] = 0;
	kk.kern[0][2] = -1;
	kk.kern[1][0] = 1;
	kk.kern[1][1] = 0;
	kk.kern[1][2] = -1;
	kk.kern[2][0] = 1;
	kk.kern[2][1] = 0;
	kk.kern[2][2] = -1;
	convk[1].push_back(kk);
	fanzhuan(&convk[1][0]);
}
void network::pooling(int layerout,int layercoor)
{
	double poolCache[32][32];
	memset(poolCache, 0, sizeof(poolCache));
	int dx=0, dy=0;
	for (int convker = 0; convker < convk[layerout].size(); convker++)
	{
		for (int x = 0; x < convk[layerout][0].outputsize - poolc[layercoor].stride; x += poolc[layercoor].stride)
		{
			for (int y = 0; y < convk[layerout][0].outputsize - poolc[layercoor].stride; y += poolc[layercoor].stride)
			{
				poolCache[dx][dy] = std::max(convk[layerout][convker].output[x][y], std::max(convk[layerout][convker].output[x][y + 1], std::max(convk[layerout][convker].output[x + 1][y + 1], convk[layerout][convker].output[x + 1][y])));
				++dy;
			}
			++dx;
			dy = 0;
		}
		memcpy(&convk[layerout][convker].output,&poolCache,sizeof(poolCache));
		convk[layerout][convker].outputsize /= 2;
		for (int i = 0; i < 32; i++)
		{
			for (int j = 0; j < 32; j++)
			{
				std::cout << convk[layerout][convker].output[i][j] << " ";
				//std::cout << poolCache[i][j] << " ";
			}
			std::cout << std::endl;
		}
	}
}
void network::convolution(int layer)
{
	for (int i = 0; i < convk[layer].size(); i++)
	{
		for (int j = 0; j < convk[layer - 1].size(); j++)
		{
			for (int y = 0; y < convk[layer - 1][j].outputsize - convk[layer][i].kersize; y += convk[layer][i].stride)
			{
				for (int x = 0; x < convk[layer - 1][j].outputsize - convk[layer][i].kersize; x += convk[layer][i].stride)
				{
					//cout<<1;
					for (int b1 = 0; b1 < convk[layer][i].kersize; b1++)
					{
						for (int b2 = 0; b2 < convk[layer][i].kersize; b2++)
						{
							convk[layer][i].output[y][x] += convk[layer - 1][j].output[y + b1][x + b2] * convk[layer][i].kern[b1][b2];
						}
					}
					convk[layer][i].output[y][x] += convk[layer][i].bias;
					convk[layer][i].output[y][x] = relu(convk[layer][i].output[y][x]);
					convk[layer][i].outputsize = (convk[layer - 1][j].outputsize - convk[layer][i].kersize) / convk[layer][i].stride + 1;
					//std::cout << convk[layer][i].outputsize;
				}
			}
		}
	}
	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			std::cout << convk[1][0].output[i][j] << " ";
		}
		std::cout << std::endl;
	}
}