#include <iostream>
#include <cmath>
#include<ctime>
using namespace std;


#define IPNNUM 5     //输入层节点数
#define HDNNUM 10    //隐含层节点数
#define OPNNUM 5     //输出层节点数

//结点类，用以构成网络
class node 
{
public:  //关键字 public 确定了类成员的访问属性。在类对象作用域内，公共成员在类的外部是可访问的。您也可以指定类的成员为 private 或 protected，
	double value; //数值，存储结点最后的状态
	double *W=NULL;    //结点到下一层的权值

	void initNode(int num);//初始化函数，必须调用以初始化权值个数
	~node();	  //析构函数，释放掉权值占用内存
};

void node::initNode(int num)
{
	W = new double[num];
	srand((unsigned)time(NULL)); //srand() 初始化随机数发生器
	for (size_t i = 0; i < num; i++)//给权值赋一个随机值
	{
		W[i]= (rand() % 100)/(double)100; //rand()随机数发生器
	}
}

node::~node()
{
	if (W!=NULL)
	{
		delete[]W;
	}
}

//网络类，描述神经网络的结构并实现前向传播以及后向传播
class net 
{
public:
	node inlayer[IPNNUM]; //输入层
	node hidlayer[HDNNUM];//隐含层
	node outlayer[OPNNUM];//输出层

	double yita = 0.1;//学习率η
	double k1;//输入层偏置项权重
	double k2;//隐含层偏置项权重
	double Tg[OPNNUM];//训练目标
	double O[OPNNUM];//网络实际输出

	net();//构造函数，用于初始化各层和偏置项权重
	double sigmoid(double z);//激活函数
	double getLoss();//损失函数，输入为目标值
	void forwardPropagation(double *input);//前向传播,输入为输入层节点的值
	void backPropagation(double *T);//反向传播，输入为目标输出值
	void printresual(int trainingTimes);//打印信息
};  

net::net()
{
	//初始化输入层和隐含层偏置项权值，给一个随机值
	srand((unsigned)time(NULL));
	k1= (rand() % 100) / (double)100;
	k2 = (rand() % 100) / (double)100;
	//初始化输入层到隐含层节点个数
	for (size_t i = 0; i < IPNNUM; i++)
	{
		inlayer[i].initNode(HDNNUM);
	}
	//初始化隐含层到输出层节点个数
	for (size_t i = 0; i < HDNNUM; i++)
	{
		hidlayer[i].initNode(OPNNUM);
	}
}
//激活函数
double net::sigmoid(double z)
{
	return 1/(1+ exp(-z));
}
//损失函数
double net::getLoss()
{
	double mloss = 0;
	for (size_t i = 0; i < OPNNUM; i++)
	{
		mloss += pow(O[i] - Tg[i], 2);
	}
	return mloss / OPNNUM;
}
//前向传播
void net::forwardPropagation(double *input)
{
	for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)//输入层节点赋值
	{
		inlayer[iNNum].value = input[iNNum];
	}
	for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)//算出隐含层结点的值
	{
		double z = 0;
		for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)
		{
			z+= inlayer[iNNum].value*inlayer[iNNum].W[hNNum];
		}
		z+= k1;//加上偏置项
		hidlayer[hNNum].value = sigmoid(z);
	}
	for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)//算出输出层结点的值
	{
		double z = 0;
		for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)
		{
			z += hidlayer[hNNum].value*hidlayer[hNNum].W[oNNum];
		}
		z += k2;//加上偏置项
		O[oNNum] = outlayer[oNNum].value = sigmoid(z);
	}
}
//反向传播，这里为了公式好看一点多写了一些变量作为中间值
//计算过程用到的公式在博文中已经推导过了，如果代码没看明白请看看博文
void net::backPropagation(double *T)
{	
	for (size_t i = 0; i < OPNNUM; i++)
	{
		Tg[i] = T[i];
	}
	for (size_t iNNum = 0; iNNum < IPNNUM; iNNum++)//更新输入层权重
	{
		for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)
		{
			double y = hidlayer[hNNum].value;
			double loss = 0;
			for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
			{
				loss += (O[oNNum] - Tg[oNNum])*O[oNNum] * (1 - O[oNNum])*hidlayer[hNNum].W[oNNum];
			}
			inlayer[iNNum].W[hNNum] -= yita*loss*y*(1 - y)*inlayer[iNNum].value;
		}
	}
	for (size_t hNNum = 0; hNNum < HDNNUM; hNNum++)//更新隐含层权重
	{
		for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
		{
			hidlayer[hNNum].W[oNNum]-= yita*(O[oNNum] - Tg[oNNum])*
				O[oNNum] *(1- O[oNNum])*hidlayer[hNNum].value;
		}
	}
}

void net::printresual(int trainingTimes)
{
	double loss = getLoss();
	cout << "训练次数：" << trainingTimes << endl;
	cout << "loss：" << loss << endl;
	for (size_t oNNum = 0; oNNum < OPNNUM; oNNum++)
	{
		cout << "输出" << oNNum+1<< "：" << O[oNNum] << endl;
	}
}

void main()
{
	net mnet;
	double minput[IPNNUM] = { 0.1, 0.2, 0.3, 0.4, 0.5 };
	double mtarget[IPNNUM] = { 0.2, 0.4, 0.6, 0.8, 1 };
	for (size_t i = 0; i < 10000; i++)
	{
		mnet.forwardPropagation(minput);//前向传播
		mnet.backPropagation(mtarget);//反向传播
		if (i%1000==0)
		{
			mnet.printresual(i);//信息打印
		}
	}
}
