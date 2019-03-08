#include <iostream>
#include <cmath>
using namespace std;

//结点类，用以构成网络
class node
{
  public:
        double value; //数值，存储结点最后的状态，对应到文章示例为X1，Y1等值
        double W[2];  //结点到下一层的权值     
};

//网络类，描述神经网络的结构并实现前向传播以及后向传播
//这里为文章示例中的三层网络，每层结点均为两个
class net
{
  public:
        node input_layer[2];//输入层结点
        node hidden_layer[2];//隐含层结点
        node output_layer[2];//输出层结点,这里只是两个数，但这样做方便后面的扩展

        double yita = 0.5;//学习率η
        double k1;//输入层偏置项权重
        double k2;//隐含层偏置项权重
        double Tg[2];//训练目标 
        double O[2];//网络实际输出

        net();//构造函数，用于初始化权重，一般可以随机初始化
        double sigmoid(double z);//激活函数
        double getLoss();//损失函数，输入为目标值
        void forwardPropagation(double input1,double input2);//前向传播
        void backPropagation(double T1, double T2);//反向传播，更新权值
        void printresual();//打印信息
};

net::net()  //表示net()是属于class net的
{
  k1 = 0.35;
  k2 = 0.60;
  input_layer[0].W[0] = 0.15;  //X1
  input_layer[0].W[1] = 0.25;
  input_layer[1].W[0] = 0.20;  //X2
  input_layer[1].W[1] = 0.30;
  hidden_layer[0].W[0] = 0.40; //Y1
  hidden_layer[0].W[1] = 0.50;
  hidden_layer[1].W[0] = 0.45; //Y2
  hidden_layer[1].W[1] = 0.55;
}

//激活函数
double net::sigmoid(double z)
{
  return 1/(1+exp(-z));
}
//损失函数
double net::getLoss()
{
    return (pow(O[0] -Tg[0],2)+ pow(O[1] - Tg[1],2))/2; //pow（x，y）求的是x的y次方
}
//前向传播
void net::forwardPropagation(double input1, double input2)
{
   input_layer[0].value = input1;
   input_layer[1].value = input2;
   for (size_t hNNum = 0; hNNum < 2; hNNum++)//算出隐含层结点的值两个节点
   {
     double z = 0;
     for (size_t iNNum = 0; iNNum < 2; iNNum++) //算出隐含层结点的值两个节点
     {
       z+= input_layer[iNNum].value*input_layer[iNNum].W[hNNum];  
     }
     z+= k1;//加上偏置项
     hidden_layer[hNNum].value = sigmoid(z);
   }

   for (size_t outputNodeNum = 0; outputNodeNum < 2; outputNodeNum++)//算出输出层结点的值
   {
        double z = 0;
        for (size_t hNNum = 0; hNNum < 2; hNNum++)
        {
            z += hidden_layer[hNNum].value*hidden_layer[hNNum].W[outputNodeNum];
        }
        z += k2;//加上偏置项
        O[outputNodeNum] = output_layer[outputNodeNum].value = sigmoid(z);
    }
}
//反向传播，这里为了公式好看一点多写了一些变量作为中间值
//计算过程用到的公式在博文中已经推导过了，如果代码没看明白请看看博文
void net::backPropagation(double T1, double T2)
{   
    Tg[0] = T1;
    Tg[1] = T2;
    for (size_t iNNum = 0; iNNum < 2; iNNum++)//更新输入层权重
    {
        for (size_t wnum = 0; wnum < 2; wnum++)
        {
            double y = hidden_layer[wnum].value;
            input_layer[iNNum].W[wnum] -= yita*((O[0] - T1)*O[0] *(1- O[0])*
                hidden_layer[wnum].W[0] +(O[1] - T2)*O[1] *(1 - O[1])*hidden_layer[wnum].W[1])*
                y*(1- y)*input_layer[iNNum].value; //待研究
        }
    }
    for (size_t hNNum = 0; hNNum < 2; hNNum++)//更新隐含层权重
    {
        for (size_t wnum = 0; wnum < 2; wnum++)
        {
            hidden_layer[hNNum].W[wnum]-= yita*(O[wnum] - Tg[wnum])*
                O[wnum] *(1- O[wnum])*hidden_layer[hNNum].value; //待研究
        }
    }
}

void net::printresual()
{
    double loss = getLoss();
    cout << "loss：" << loss << endl;
    cout << "输出1：" << O[0] << endl;
    cout << "输出2：" << O[1] << endl;
}

int main()   //void
{
    net mnet;  //创建mnet实例
    for (size_t i = 0; i < 10000; i++)
    {
        mnet.forwardPropagation(0.05, 0.1);//前向传播
        mnet.backPropagation(0.01, 0.99);//反向传播
        if (i%1000==0)
        {
            mnet.printresual();//反向传播
        }
    }
}
