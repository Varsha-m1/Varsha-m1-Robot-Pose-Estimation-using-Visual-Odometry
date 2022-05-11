import glob
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Function
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
import math

def get_image(path, img_size=(1280,384)):
	img = cv2.imread(path)
	img = cv2.resize(img, img_size, cv2.INTER_LINEAR)
	return img

def load_images(img_dir, img_size):
	print("images", img_dir)
	images = []
	images_set = []
	for img in glob.glob(img_dir+'/*'):
		images.append(get_image(img,img_size))
	for i in range(len(images)-1):
		img1 = images[i]
		img2 = images[i+1]
		img = np.concatenate([img1,img2],axis=-1)
		images_set.append(img)
	print("images count: ", len(images_set))
	images_set = np.reshape(images_set, (-1,6,384,1280))
	return images_set

def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt,R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I * shouldBeIdentity)
	return n<1e-6

def rotationMatrixToEulerAngles(R):
	assert(isRotationMatrix(R))
	sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
	singular = sy<1e-6
	if not singular:
		x = math.atan2(R[2,1], R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	return np.array([x,y,z])


def getMatrices(all_poses):
	all_matrices = []
	for i in range(len(all_poses)):
		j = all_poses[i]
		p = np.array(j[3],j[7],j[11])
		R = np.array([j[0],j[1],j[2]],[j[4],j[5],j[6]],[j[8],j[9],j[10]])
		angles = rotationMatrixToEulerAngles(R)
		matrix = np.concatenate((p, angles))
		all_matrices.append(matrix)
	return all_matrices

def load_poses(pose_file):
	print("pose ", pose_file)
	poses = []
	poses_set = []
	with open(pose_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			pose = np.fromstring(line, dtype = float, sep = ' ')
			poses.append(pose)
	poses = getMatrices(poses)
	for i in range(len(poses)-1):
		pose1 = poses[i]
		poses2 = poses[i+1]
		finalposes = pose2-pose1
		poses_set.append(finalpose)
	print("poses count ", len(poses_set))
	return poses_set

def VODataLoader(datapath, img_size=(1280,384), test=False):
	print (datapath)
	poses_path = os.path.join(datapath, 'dataset/poses')
	img_path = os.path.join(datapath, 'dataset/sequences')
	if test:
		sequences = ['03']
	else:
		sequences = ['01']
	images_set = []
	odometry_set = []
	for sequence in sequences:
		images_set.append(torch.FloatTensor(load_images(os.path.join(img_path, sequence, 'image_1'),img_size)))
		odometry_set.append(torch.FloatTensor(load_poses(os.path.join(poses_path,sequence+'.txt'))))
	return images_set, odometry_set

X, y = VODataLoader("/home/loki/VO")

class DeepVONet(nn.Module):
	def __init__(self):
		super(DeepVONet, self).__init__()
		self.conv1 = nn.Conv2d(6,64,kernel_size=(7,7),stride=(2,2),padding=(3,3))
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(64,128,kernel_size=(5,5),stride=(2,2),padding=(2,2))
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(128,256,kernel_size=(5,5),stride=(2,2),padding=(2,2))
		self.relu3 = nn.ReLU(inplace=True)
		self.conv3_1 = nn.Conv2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1))
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv4 = nn.Conv2d(256,512,kernel_size=(3,3),stride=(2,2),padding=(1,1))
		self.relu4 = nn.ReLU(inplace=True)
		self.conv4_1 = nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1))
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv5 = nn.Conv2d(512,512,kernel_size=(3,3),stride=(2,2),padding=(1,1))
		self.relu5 = nn.ReLU(inplace=True)
		self.conv5_1 = nn.Conv2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1))
                self.relu5_1 = nn.ReLU(inplace=True)
		self.conv6 = nn.Conv2d(512,1024,kernel_size=(3,3),stride=(2,2),padding=(1,1))
		self.lstm1 = nn.LSTMCell(20*6*1024, 100)
		self.lstm2 = nn.LSTMCell(100,100)
		self.fc = nn.Linear(in_features=100, out_features=6)
		self.reset_hidden_states()

	def reset_hidden_states(self, size=10, zero=True):
		if zero:
			self.hx1 = Variable(torch.zeros(size, 100))
			self.cx1 = Variable(torch.zeros(size, 100))
			self.hx2 = Variable(torch.zeros(size, 100))
			self.cx2 = Variable(torch.zeros(size, 100))
		else:
			self.hx1 = Variable(self.hx1.data)
			self.cx1 = Variable(self.cx1.data)
			self.hx2 = Variable(self.hx2.data)
			self.cx2 = Variable(self.cx2.data)
		if next(self.parameters()).is_cuda == True:
			self.hx1 = self.hx1.cuda()
			self.cx1 = self.cx1.cuda()
			self.hx2 = self.hx2.cuda()
			self.cx2 = self.cx2.cuda()

	def forward(self,x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.conv3_1(x)
		x = self.relu3_1(x)
		x = self.conv4(x)
		x = self.relu4(x)
		x = self.conv4_1(x)
		x = self.relu4_1(x)
		x = self.conv5(x)
		x = self.relu5(x)
		x = self.conv5_1(x)
		x = self.conv6(x)
		x = x.view(x.size(0), 20*6*1024)
		self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))
		x = self.hx1
		self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))
		x = self.hx2
		x = self.fc(x)
		return x

	
def testing_model (model, test_num, X):
	start_time = time.time()
	Y_output = []
	count, totcount = 0,0
	for i in range(test_num):
		inputs = X[i]
		outputs = model(inputs)
		Y_output.append(outputs)
	print("Time taken for testing {0}".format((time.time() - start_time)))
	return torch.stack(Y_output)
	

def get_accuracy (outputs, labels, batch_size):
	diff = 0
	for i in range(batch_size):
		for j in range(10):
			out = outputs[i][j].detach().numpy()
			lab = labels[i][j].numpy()
			diff += get_mse(out, lab)
	print("Accuracy: ", (1-diff/(batch_size*10))*100, "%")

def get_mse_diff(x,y):
	diff = 0
	for i in range(6):
		diff += (x[i]-y[i])**2
	return diff/6

model = DeepVONet()
print(model)

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5, weight_decay=0.5)

X_test, y_test = VODataLoader("/home/loki/VO", test=True)
X_test = torch.stack(X_test).view(-1,10,6,384,1280)
y_test = torch.stack(y_test).view(-1,10,6)
print(X_test.size())
print(y_test.size())
y_output = testing_model(model, test_batch_size, X)
print(y_output.size())
torch.save(y_output, "y_outputSeq01Img01.pt")
get_accuracy(y_output, y_test, test_batch_size)

