import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))




# get words polarity
words_polarity = {}
with open("EmotionLookupTable.txt", "r") as ins:
        for line in ins:
            items = line.split()
            word = items[0]
            word = word.replace('*','')

            score = int(items[1])

            words_polarity[word] = score



# train sentences
X1=[]
Y1=[]

with open("pos.txt", "r") as ins:
        for line in ins:
            

            words = line.split()

            total_pos_score = 0
            total_neg_score = 0

            count_pos_words = 0
            count_neg_words = 0

            for w in words:
                if words_polarity.has_key(w):
                    score = words_polarity[w]
                    if score>0:
                        total_pos_score += score
                        count_pos_words +=1
                    else:
                        total_neg_score += (-1*score)
                        count_neg_words +=1
            
            # if count_pos_words==0 and count_neg_words==0:
            #     continue

            x = [total_pos_score, total_neg_score, count_pos_words, count_neg_words ]
            y = [1]
            X1.append(x);
            Y1.append(y);


X2=[]
Y2=[]

with open("neg.txt", "r") as ins:
        for line in ins:
            

            words = line.split()

            total_pos_score = 0
            total_neg_score = 0

            count_pos_words = 0
            count_neg_words = 0

            for w in words:
                if words_polarity.has_key(w):
                    score = words_polarity[w]
                    if score>0:
                        total_pos_score += score
                        count_pos_words +=1
                    else:
                        total_neg_score += (-1*score)
                        count_neg_words +=1

            x = [total_pos_score, total_neg_score, count_pos_words, count_neg_words ]
            y = [0]
            X2.append(x);
            Y2.append(y);


shuffled_x = []
shuffled_y = []
for i in range(0,5331):
    shuffled_x.append(X1[i])
    shuffled_y.append(Y1[i])
    shuffled_x.append(X2[i])
    shuffled_y.append(Y2[i])







# train
alpha = 0.001

index = int(len(shuffled_x)*0.8)
x_train = np.array(shuffled_x[0:index])               
y_train = np.array(shuffled_y[0:index])


num_input_neurons=4
num_hidden_neurons=5
num_output_neurons=1

syn0 = 2*np.random.random((num_input_neurons,num_hidden_neurons)) - 1
syn1 = 2*np.random.random((num_hidden_neurons,num_output_neurons)) - 1

for j in xrange(60000):

	
    l0 = x_train
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    l2_error = y_train - l2
    l2_delta = l2_error*nonlin(l2,deriv=True)

   
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += alpha*l1.T.dot(l2_delta)
    syn0 += alpha*l0.T.dot(l1_delta)





#train accuracy by perceptron
num_total = len(x_train)
num_correct = 0

i=0
for x in x_train:
    y = y_train[i][0]

    l1 = nonlin(np.dot(x,syn0))
    l2= nonlin(np.dot(l1,syn1))

    # predicted_y = -1
    # if l2<0.5:
    #     predicted_y = 0
    # else:
    #     predicted_y = 1

    if (l2>=0.4 and y==1 ) or (l2<0.6 and y==0 ):
        num_correct+=1

    i+=1

train_accuracy = float(num_correct/float(num_total))*100
print "train_accuracy: "+str(train_accuracy)


#test accuracy by perceptron

x_test = shuffled_x[index:5331*2]
y_test = shuffled_y[index:5331*2]

 
num_total = len(x_test)
num_correct = 0

i=0
for x in x_test:
    y = y_test[i][0]

    l1 = nonlin(np.dot(x,syn0))
    l2= nonlin(np.dot(l1,syn1))

    # predicted_y = -1
    # if l2<0.5:
    #     predicted_y = 0
    # else:
    #     predicted_y = 1

    if (l2>=0.4 and y==1 ) or (l2<0.6 and y==0 ):
        num_correct+=1

    i+=1

test_accuracy = float(num_correct/float(num_total))*100
print "test_accuracy: "+str(test_accuracy)




#test accuracy for normal approach

x_test = shuffled_x[index:5331*2]
y_test = shuffled_y[index:5331*2]

 
num_total = len(x_test)
num_correct = 0

i=0
for x in x_test:
    y = y_test[i][0]

    sum_pos_score = x[0]
    sum_neg_score = x[1]

    if (sum_pos_score>=sum_neg_score and y==1 ) or (sum_pos_score<sum_neg_score and y==0 ):
        num_correct+=1

    i+=1

test_accuracy = float(num_correct/float(num_total))*100
print "test_accuracy for normal approach: "+str(test_accuracy)
