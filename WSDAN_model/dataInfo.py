from dataset import *
import pickle
import csv

train_dataset = INAT("../../dataset","../../dataset/train2019.json",(512,512),is_train=True)
num_classes = len(set(train_dataset.classes))
#print(set(train_dataset.classes))

class_count = []
for i in range(num_classes):
    class_count.append(train_dataset.classes.count(i))
total = sum(class_count)

print(class_count)
print(total)


weights = []
for i in range(len(train_dataset.imgs)):
    weights.append(1/class_count[train_dataset.classes[i]])

pickle.dump(class_count,open("class_count.pkl","wb"))


########################################################################################
"""
val_dataset = INAT("../../dataset","../../dataset/val2019.json",(512,512),is_train=False)
num_classes_val = len(set(val_dataset.classes))

class_count_val = []
for i in range(num_classes_val):
    class_count_val.append(val_dataset.classes.count(i))

print(class_count_val)
print(sum(class_count_val))


test_dataset = INAT("../../dataset","../../dataset/test2019.json",(512,512),is_train=False)
num_classes_test = len(set(test_dataset.classes))

class_count_test = []
for i in range(num_classes_test):
    class_count_test.append(test_dataset.classes.count(i))

print(class_count_test)
print(sum(class_count_test))
"""
