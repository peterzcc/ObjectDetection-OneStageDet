image_names = open('train.txt', 'r').readlines()
image_name = [item.strip() for item in image_names]
#print(image_name)
N = len(image_name)
print('There are %s images in the training set.'%N)
# output:
# N: 5717


# calculate how many images contain a specific class object.

classes = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}
num = {}
summation = 0
for c in classes:
	n = 0
	lines = open('%s_train.txt'%(c), 'r').readlines()
	for line in lines:
		sample = line.strip()
		id = sample.index(' ')
		label = sample[id+1:]
		if not label == '-1':
			n = n + 1
	num[c] = n
	summation = summation + n
print(num)
print(summation)

# output:
# num:
# {'sheep': 171, 'horse': 238, 'bicycle': 281, 'bottle': 399, 'cow': 155, 'sofa': 359, 'motorbike': 274, 'dog': 636, 'cat': 540, 'person': 2142, 'train': 275, 'boat': 264, 'aeroplane': 328, 'bus': 219, 'pottedplant': 289, 'tvmonitor': 299, 'chair': 656, 'bird': 399, 'diningtable': 318, 'car': 621}
# summation:
# 8863