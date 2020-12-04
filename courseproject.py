from sklearn.metrics import accuracy_score
from sklearn import svm
import random
import statistics
import sys

##import files

print('Train Data and Train Labels import in progress:')

train_datafile = sys.argv[1]
f = open(train_datafile,'r')
train_data = []
x = f.readline()

while(x!=''):
    a = x.split()
    y = []
    for j in range(0, len(a), 1):
        y.append(float(a[j]))
    train_data.append(y)
    x = f.readline()
    
row_count = len(train_data)
column_count = len(train_data[0])

train_labelfile = sys.argv[2]
f = open(train_labelfile)
train_labels = {}
num = [0,0]
x = f.readline()
while(x != ''):
    a = x.split()
    train_labels[int(a[1])] = int(a[0])
    x = f.readline()
    num[int(a[0])] += 1

#print(train_labels)
print('Completed.')
print('Test Data import in progress:')

test_datafile = sys.argv[3]
f = open(test_datafile,'r')
test_data = []
x = f.readline()

while(x!=''):
    b = x.split()
    z = []
    for j in range(0, len(b), 1):
        z.append(float(b[j]))
    test_data.append(z)
    x = f.readline()

row_count_test = len(test_data)
print('Completed.')
print('F-scores computation in progress:')

#Compute Mean for all features
mean_0 = []
mean_1 = []
mean = []
for j in range(0, column_count, 1):
    mean_0.append(0)
    mean_1.append(0)
    mean.append(0)

for i in range(0, row_count, 1):
    if(train_labels.get(i) != None and train_labels[i] == 0):
        for j in range(0, column_count, 1):
            mean_0[j] = mean_0[j] + train_data[i][j]
            
    if(train_labels.get(i) != None and train_labels[i] == 1):
        for j in range(0, column_count, 1):
            mean_1[j] = mean_1[j] + train_data[i][j]
            
for j in range(0, column_count, 1):
    mean_0[j] = mean_0[j]/num[0]
    mean_1[j] = mean_1[j]/num[1]

for i in range(column_count):
    mean[i] = (mean_0[i] + mean_1[i])/2
    
#Compute F-scores
f_scores = [0]*column_count
for i in range(column_count):
    a = (mean_1[i] - mean[i]) ** 2 + (mean_0[i] - mean[i]) ** 2
    
    b = 0
    c = 0
    for k in range(0, row_count, 1):
        if(train_labels.get(k) != None and train_labels[k] == 0):
            c = c + (train_data[k][i] - mean_0[i]) ** 2
        elif(train_labels.get(k) != None and train_labels[k] == 1):
            b = b + (train_data[k][i] - mean_1[i]) ** 2
            
    b = b / (num[1] - 1)
    c = c / (num[0] - 1)
    
    f_scores[i] = a /(b+c)

mean_fscores = statistics.mean(f_scores)
std_fscores = statistics.stdev(f_scores)

print('F-scores are computed for all the features')
#Picking up threshold and dropping features

threshold = mean_fscores + 19 * std_fscores
#print(threshold)
new_train_data = []
column_list = []
new_test_data = []

for i in range(0, row_count, 1):
    temp = []
    for j in range(0, column_count, 1):
        if(f_scores[j] > threshold):
            temp.append(train_data[i][j])
            column_list.append(j)
    new_train_data.append(temp)
    
for i in range(0, row_count_test, 1):
    temp1 = []
    for j in range(0, column_count, 1):
        if(f_scores[j] > threshold):
            temp1.append(test_data[i][j])
    new_test_data.append(temp1)

            
new_column_count = len(new_train_data[0])
final_feature_list = set(column_list)
# final_feature_list.sort()

print("The total number of features selected is: ",new_column_count)
print("The features' column indices are:", sorted(final_feature_list))
print('Data with above features extracted from Train Data')

#Converting train labels dict to list
train_label= list(train_labels.values())

#SVM classification

print('Performing SVM by training on Train Data & Train Labels and predicting on Test Data')        
clf = svm.SVC(gamma='scale')
clf.fit(new_train_data,train_label)
test_labels = list(clf.predict(new_test_data))

#Printing the predictions

print('The predictions are: ')

for o in range(len(test_labels)):
    print(test_labels[o],o)

   
        
    