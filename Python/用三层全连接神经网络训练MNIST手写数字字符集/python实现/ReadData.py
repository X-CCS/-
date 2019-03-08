import numpy as np
import struct

def loadImageSet(filename):
	print ("load image set",filename)
	binfile= open(filename, 'rb')
	buffers = binfile.read()
	head = struct.unpack_from('>IIII' , buffers ,0)
	print ("head,",head)
	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]
	#[60000]*28*28
	bits = imgNum * width * height
	bitsString = '>' + str(bits) + 'B' #读取定长数据段，即字符集图片总和
	imgs = struct.unpack_from(bitsString,buffers,offset)
	binfile.close()
	imgs = np.reshape(imgs,[imgNum,1,width*height])#将字符集图片分隔为单张图片
	print ("load imgs finished")
	return imgs

def loadLabelSet(filename):
	print ("load label set",filename)
	binfile = open(filename, 'rb')
	buffers = binfile.read()
	head = struct.unpack_from('>II' , buffers ,0)
	print ("head,",head)
	imgNum=head[1]
	offset = struct.calcsize('>II')
	numString = '>'+str(imgNum)+"B"
	labels = struct.unpack_from(numString , buffers , offset)
	binfile.close()
	labels = np.reshape(labels,[imgNum,1])
	print ('load label finished')
	return labels
