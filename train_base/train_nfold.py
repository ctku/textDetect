#Train by n-fold cross-validation
################################################################################
# using libSVM
# -s svm_type : set type of SVM (default 0)
#	0 -- C-SVC
#	1 -- nu-SVC
#	2 -- one-class SVM
# -t kernel_type : set type of kernel function (default 2)
#	0 -- linear: u'*v
#	1 -- polynomial: (gamma*u'*v + coef0)^degree
#	2 -- radial basis function: exp(-gamma*|u-v|^2)
# -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
#              set higher weight to minority class
################################################################################

def experiment_libsvm(fold, data, id_map, params):
    # fold: fold for cross-validation
    # id_map: mapping from index of a node to its attribute ID
    # params: specify parameters used in experiment
    # random sampling testing and training data given number of pos and neg training and testing samples
    # ID of data[k] = id_map[k]
    ave_accuracy = 0
    ave_precision = 0
    ave_recall = 0

    false_neg_node = [0]*len(data)
    false_pos_node = [0]*len(data)

    f1 = open(false_pos_file,'w')
    f2 = open(false_neg_file,'w')
    f3 = open(result_file_name,'w')
   # f4 = open(svm_weight_file,'w') # summation of weights from different folds of training using linear SVM
    w1 = csv.writer(f1,delimiter = '\n')
    w2 = csv.writer(f2,delimiter = '\n')
    w3 = csv.writer(f3,delimiter = '\t')
  #  w4 = csv.writer(f4,delimiter = '\n')
    
    pos_data_idx = [id for id in range(len(data)) if data[id][0][0]=='1'] # error0
    neg_data_idx = [id for id in range(len(data)) if data[id][0][0]=='0'] # not error
    
    if params[0] == 1:
        train_num_pos = params[1]
        train_num_neg = params[2]
        test_num_pos = params[3]
        test_num_neg = params[4]
        ww0 = params[5]
        ww1 = params[6]
    elif params[0] == 2:
        num_train_neg = params[1]
        ww0 = params[2]
        ww1 = params[3]
        # split data into different folds
        fold_idx = [[]]*fold
        nn = int(len(data)/fold)
        for j in range(fold):
            fold_idx[j] = range(j*nn,(j+1)*nn)        
    
    # weight = [0] * fv_num
    
    for ii in range(fold):
        print 'fold %d:\n' % ii
        if params[0] == 1:
            # test_ID = {k: data[k] in test set}
            test_ID = data_preparation_random(data, train_num_pos, train_num_neg, test_num_pos, test_num_neg, pos_data_idx, 
neg_data_idx) 
        elif params[0] == 2:
            test_ID = data_preparation_v2(data, fold_idx, num_train_neg, ii)
            
        print 'start svm training\n'
        svm_cmd1 = './libsvm/svm-scale -l 0 -u 1 -s range %s > %s' % (train_file_name, train_file_name+'.scale')
        os.system(svm_cmd1)
        
        svm_cmd2 = './libsvm/svm-scale -r range %s > %s' % (test_file_name, test_file_name+'.scale')
        os.system(svm_cmd2)

        svm_cmd3 = './libsvm/svm-train -s 0 -t 0 -w1 %d -w0 %d %s' % (ww1, ww0, train_file_name+'.scale')
        os.system(svm_cmd3)
        
        # find_SVM_weight(fv_num,train_file_name+'.scale.model',weight)
        
        print 'start svm testing\n'
        svm_cmd4 = './libsvm/svm-predict %s %s %s' % (test_file_name+'.scale', train_file_name+'.scale.model', 
svm_output_file_name)
        os.system(svm_cmd4)
        
        # evaluation
        predict_label = [int(w[0][0]) for w in csv.reader(open(svm_output_file_name,'rb'))]
        true_label = [int(w[0][0]) for w in csv.reader(open(test_file_name,'rb'))]
        
        N = len(predict_label)
        
        true_pos =[i for i in range(N) if predict_label[i]==true_label[i] and predict_label[i]==1]
        false_pos = [i for i in range(N) if predict_label[i]!=true_label[i] and predict_label[i]==1]
        true_neg = [i for i in range(N) if predict_label[i]==true_label[i] and predict_label[i]==0]
        false_neg = [i for i in range(N) if predict_label[i]!=true_label[i] and predict_label[i]==0]
        accuracy = (len(true_pos)+len(true_neg))/float(N)
        precision = len(true_pos)/float(len(true_pos)+len(false_pos))
        recall = len(true_pos)/float(len(true_pos)+len(false_neg))
        print "accuracy = %f, precision = %f, recall = %f" % (accuracy, precision, recall)
        ave_accuracy = ave_accuracy + accuracy
        ave_precision = ave_precision + precision
        ave_recall = ave_recall+ recall
        # find false nodes
        for id in false_pos:
            false_pos_node[test_ID[id]] +=1
        
        for id in false_neg:
            false_neg_node[test_ID[id]] +=1
        
        ID_pred_pair = [[id_map[test_ID[i]],predict_label[i]] for i in range(N)]
        w3.writerows(ID_pred_pair)
           
    ave_accuracy = ave_accuracy/fold
    ave_precision = ave_precision/fold
    ave_recall = ave_recall/fold
    ave_F = 2*ave_precision*ave_recall/(ave_precision+ave_recall)
    print "Average: accuracy = %f, precision = %f, recall = %f" % (ave_accuracy, ave_precision, ave_recall)
    false_pos_ID = [id_map[id] for id in range(len(data)) if false_pos_node[id]>0]
    false_neg_ID = [id_map[id] for id in range(len(data)) if false_neg_node[id]>0]
    
    print "number of false positive nodes = %d \n" % len(false_pos_ID)
    print "number of false negative nodes = %d \n" % len(false_neg_ID)
    #print "search %f of all nodes can find %f errors\n" % (float(len(false_pos_ID)+len(true_pos_ID))/len(data), 
ave_recall)
   # f4_data = ["%s:%s" % (i+1,weight[i]) for i in range(len(weight))]  
    w1.writerow(false_pos_ID)
    w2.writerow(false_neg_ID)
   # w4.writerow(f4_data)
    f1.close()
    f2.close()
    f3.close()
    #f4.close()

return [ave_accuracy, ave_precision, ave_recall, ave_F]
