#include "setting.h"
#include "net.hpp"//神经网络
#include "getImg.hpp"//训练数据

void AccuracyRate(int time, net *mnet, getImg *mImg)//精确率评估
{
	double tagright = 0;//正确个数统计
	for (size_t count = 0; count < 10000; count++)
	{
		mnet->forwardPropagation(mImg->mImgData[count].data);//前向传播
		double value = -100;
		int gettag = -100;
		for (size_t i = 0; i < 10; i++)
		{
			if (mnet->outlayer[i].value > value)
			{
				value = mnet->outlayer[i].value;
				gettag = i;
			}
		}
		if (mImg->mImgData[count].tag == gettag)
		{
			tagright++;
		}
	}
	//mnet.printresual(0);//信息打印
	cout << "第" << time + 1 << "轮:  ";
	cout << "正确率为:" << tagright / 10000 << endl;
}

int main()
{
	getImg mGetTrainImg;
	mGetTrainImg.imgTrainDataRead("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	getImg mGetTestImg;
	mGetTestImg.imgTrainDataRead("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	net mnet;//神经网络对象
	for (size_t j = 0; j < 10; j++)
	{
		for (size_t i = 0; i < 60000; i++)
		{
			mnet.forwardPropagation(mGetTrainImg.mImgData[i].data);//前向传播
			mnet.backPropagation(mGetTrainImg.mImgData[i].label);//反向传播
		}
		AccuracyRate(j,&mnet, &mGetTestImg);
	}
	std::cout << "搞完收工!\n"; 
}