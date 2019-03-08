#pragma once
#include "setting.h"

class ImgData//单张图像
{
public:
	unsigned char tag;
	double data[IPNNUM];
	double label[OPNNUM];
};

class getImg
{
public:
	ImgData* mImgData;
	void imgTrainDataRead(const char *datapath, const char *labelpath);
	~getImg();
};

void getImg::imgTrainDataRead(const char *datapath, const char *labelpath)
{
	/***********读取图片数据***********/
	unsigned char readbuf[4];//信息数据读取空间
	FILE *f;
	fopen_s(&f, datapath, "rb");
	fread_s(readbuf, 4, 1, 4, f);//读取魔数，即文件标志位
	fread_s(readbuf, 4, 1, 4, f);//读取数据集图像个数
	int sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
	fread_s(readbuf, 4, 1, 4, f);//读取数据集图像行数
	int imgheight = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像行数
	fread_s(readbuf, 4, 1, 4, f);//读取数据集图像列数
	int imgwidth = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像列数
	mImgData = new ImgData[sumOfImg];
	unsigned char *data = new unsigned char[IPNNUM];
	for (int i = 0; i < sumOfImg; i++)
	{
		fread_s(data, IPNNUM, 1, IPNNUM, f);//读取数据集图像列数
		for (size_t px = 0; px < IPNNUM; px++)//图像数据归一化
		{
			mImgData[i].data[px] = data[px]/(double)255*0.99+0.01;
		}
	}
	delete[]data;
	fclose(f);
	/**********************************/
   /***********读取标签数据***********/
   /**********************************/
	fopen_s(&f, labelpath, "rb");
	fread_s(readbuf, 4, 1, 4, f);//读取魔数，即文件标志位
	fread_s(readbuf, 4, 1, 4, f);//读取数据集图像个数
	sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
	for (int i = 0; i < sumOfImg; i++)
	{
		fread_s(&mImgData[i].tag, 1, 1, 1, f);//读取数据集图像列数
		for (size_t j = 0; j < 10; j++)
		{
			mImgData[i].label[j] = 0.01;
		}
		mImgData[i].label[mImgData[i].tag] = 0.99;
	}
	fclose(f);
}

getImg::~getImg()
{
	delete[]mImgData;
}
