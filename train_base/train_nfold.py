def svmExperiment(fv_set,fv_ID,fv_name,file_name,out_name,v):
    # fv_set: a list of list, consist of feature vectors
    # fv_ID: node ID of features
    # v: fold of cross-validation
    # file_name: name of the file of extracted features
    # fv_name = [w for w in uni_desc_set] + [w for w in bi_desc_set] + ['class']
    # out_name: name of output file consisting of classification result for each node
    fv1 = [w for w in fv_set if w[-1]==1]
    fv2 = [w for w in fv_set if w[-1]==-1]
    ID1 = [fv_ID[i] for i in range(len(fv_ID)) if fv_set[i][-1]==1]
    ID2 = [fv_ID[i] for i in range(len(fv_ID)) if fv_set[i][-1]==-1]
    step1 = len(fv1)/v
    step2 = len(fv2)/v
    precision = 0
    recall = 0
    accuracy = 0
    out_ID = []  # output node ID
    out_TC = []  # output true class label 
    out_CC = []  # output classification result
    out_SS = []  # output score 
    out_FF = []  # output fold ID
    for i in range(v):
        if i<v-1:
            test_range_1 = range(step1*i,step1*(i+1))
            test_range_2 = range(step2*i,step2*(i+1))
        else:
            test_range_1 = range(step1*i,len(fv1))
            test_range_2 = range(step2*i,len(fv2))
        fv1_test = [fv1[j] for j in test_range_1]
        fv2_test = [fv2[j] for j in test_range_2]
        ID1_test = [ID1[j] for j in test_range_1]
        ID2_test = [ID2[j] for j in test_range_2]
        fv1_train = [fv1[j] for j in range(len(fv1)) if j not in test_range_1]
        fv2_train = [fv2[j] for j in range(len(fv2)) if j not in test_range_2]
        csv_test = open(file_name + '_test.csv','w')
        csv_train = open(file_name + '_train.csv','w')
        
        fv_writer = csv.writer(csv_test)
        fv_writer.writerow(fv_name)
        fv_writer.writerows(fv1_test)
        fv_writer.writerows(fv2_test)
        csv_test.close()
        CSV2SVM(file_name + '_test.csv', file_name + '_test.dat')
        
        fv_writer = csv.writer(csv_train)
        fv_writer.writerow(fv_name)
        fv_writer.writerows(fv1_train)
        fv_writer.writerows(fv2_train)
        csv_train.close()
        CSV2SVM(file_name + '_train.csv', file_name + '_train.dat')
        j = len(ID2_test)/float(len(ID1_test))
        os.system('./svm_learn -j '+str(j)+' '+ file_name + '_train.dat ' + file_name + '_model')
        os.system('./svm_classify ' + file_name + '_test.dat ' + file_name + '_model ' + file_name + '_out_v' + str(i+1))
        # read svm output
        outF = open(file_name+'_out_v'+str(i+1),'r')
        output_reader = outF.read()
        lines = output_reader.split('\n')
        prediction = [float(s) for s in lines if len(s)>0]
        cc = [int(w>0)*2-1 for w in prediction]
        out_TC = out_TC + [1]*len(ID1_test) + [-1]*len(ID2_test)
        out_CC = out_CC + cc
        out_ID = out_ID + ID1_test + ID2_test
        out_SS = out_SS + prediction
        out_FF = out_FF + [i]*len(cc)
    f = open(out_name,'w')
    out_writer = csv.writer(f,delimiter='\t')
    for i in range(len(out_CC)):
        row = [out_ID[i], out_TC[i], out_CC[i], out_SS[i], out_FF[i]]
        out_writer.writerow(row)
    f.close()


def CSV2SVM(csv_file, dat_file):
    """
    Convert a csv file into the format used in libsvm
    csv_file: each row consists of [Feature Vectors] [True Label]
    dat_file: format used in libsvm
    """
    csv_f = open(csv_file,'rb') 
    csv_reader = csv.reader(csv_f)
    csv_data = [w for w in csv_reader]  
    csv_f.close()
    csv_data = csv_data[1:]
    svm_file = open(dat_file,'wb')
    svm_writer = csv.writer(svm_file,delimiter = ' ')
    for data in csv_data:
        data = [data[-1]]+["%s:%s" % (x+1,data[x]) for x in range(len(data)-1) if float(data[x])!=0] # index starts from 1
        svm_writer.writerow(data)
    svm_file.close()
