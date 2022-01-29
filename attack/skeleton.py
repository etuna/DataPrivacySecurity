import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)

    Y_train_copy = copy.deepcopy(y_train)

    numItem = len(Y_train_copy)
    randomList = []
    class0 = 0;
    class1 = 1;
    for i in range(0, int(numItem * n)):
        randNum = random.randint(0, numItem - 1)
        if randNum not in randomList:
            randomList.append(randNum)
        else:
            while randNum in randomList:
                randNum = random.randint(0, numItem - 1)
            randomList.append(randNum)

    ### In order check flipping, remove the comments ###

    # randomList.sort()
    # print(randomList)
    # print("LEN RANDOMLIST:", len(randomList))
    # print("\n\n")

    # Flipping the classes
    for i in randomList:
        act_class = int(Y_train_copy[i]);
        if act_class == class0:
            Y_train_copy[i] = class1
        else:
            Y_train_copy[i] = class0

    # counter = 0
    # inds = []
    # for i in range(0,len(Y_train_copy)):
    #     if Y_train_copy[i] != y_train[i]:
    #         counter += 1
    #         inds.append(i)
    # print("COUNTER:", counter)
    # print("INDS:: ", inds)

    # ["DT", "LR", "SVC"]
    if model_type == "DT":
        myDEC.fit(X_train, Y_train_copy)
        DEC_predict = myDEC.predict(X_test)
        acc = accuracy_score(y_test, DEC_predict)
        # print('LF-ATTACK - DT - Accuracy of decision tree: ' + str(acc))
        return acc
    elif model_type == "LR":
        myLR.fit(X_train, Y_train_copy)
        LR_predict = myLR.predict(X_test)
        acc = accuracy_score(y_test, LR_predict)
        # print('LF-ATTACK - LR - Accuracy of logistic regression: ' + str(acc))
        return acc
    else:
        mySVC.fit(X_train, Y_train_copy)
        SVC_predict = mySVC.predict(X_test)
        acc = accuracy_score(y_test, SVC_predict)
        # print('LF-ATTACK - SVC - Accuracy of SVC: ' + str(acc))
        return acc


###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)

    X_train_copy = copy.deepcopy(X_train)
    Y_train_copy = copy.deepcopy(y_train)

    X_train_class0_inds = []
    X_train_class1_inds = []

    # print("Y_TRAINNNNNNNN", y_train)
    for i in range(len(y_train)):
        # print("iiii:::", i)
        if y_train[i] == 0:
            X_train_class0_inds.append(i)
        else:
            X_train_class1_inds.append(i)

    # print("X_train_c0:", X_train_class0_inds)
    # print("X_train_c1:", X_train_class1_inds)

    numItem = len(Y_train_copy)
    randomList = []
    for i in range(0, num_samples):
        randNum = random.randint(0, numItem - 1)
        if randNum not in randomList and Y_train_copy[randNum] == 0:
            randomList.append(randNum)
        else:
            while randNum in randomList or Y_train_copy[randNum] == 1:
                randNum = random.randint(0, numItem - 1)
            randomList.append(randNum)

    # print(randomList)
    # for i in randomList:
    #     print("ytc:",Y_train_copy[i])
    ind = 30
    while Y_train_copy[ind] != 1:
        ind += 1

    ind1 = 30;
    while Y_train_copy[ind1] != 1:
        ind1 += 1

    backdoorTrigger1 = X_train_copy[ind][0]
    backdoorTrigger2 = X_train_copy[ind][1]
    # backdoorTrigger3 = X_train_copy[ind][2]
    # backdoorTrigger4 = X_train_copy[ind][3]

    # print("TRIGGER:",backdoorTrigger)
    # print("first size:", num_samples,len(X_train_copy))
    for i in randomList:
        newRow = X_train_copy[i]
        newRow[0] = backdoorTrigger1
        # newRow[1] = backdoorTrigger2
        # newRow[2] = backdoorTrigger3
        # newRow[3] = backdoorTrigger4
        X_train_copy = np.vstack([X_train_copy, newRow])
        newYRow = Y_train_copy[ind1]
        Y_train_copy = np.append(Y_train_copy, newYRow)
        # marr = [1]
        # print("nyr:",newYRow)
        # print(Y_train_copy)
        # Y_train_copy = np.vstack([Y_train_copy, newYRow])

    # print("Secıbd size:", len(X_train_copy))
    # print("YTRRRRRRRRR",Y_train_copy)
    X_test = copy.deepcopy(X_train_copy)
    X_test = np.delete(X_test, [i for i in range(0, len(X_test))], axis=0)

    # for i in range(0, len(X_test - 1)):
    #      X_test = np.delete(X_test, i, axis=0)
    # print("is empty?:", len(X_test))
    for i in X_train_class0_inds:
        newRow = X_train_copy[i]
        newRow[0] = backdoorTrigger1
        # newRow[1] = backdoorTrigger2
        # newRow[2] = backdoorTrigger3
        # newRow[3] = backdoorTrigger4
        X_test = np.vstack([X_test, newRow])
    print("xtes:", len(X_test), X_test)
    # ["DT", "LR", "SVC"]
    if model_type == "DT":
        model_bd = myDEC.fit(X_train_copy, Y_train_copy)
        DEC_predict = myDEC.predict(X_test)
        counter1 = 0;
        for i in DEC_predict:
            if i == 1:
                counter1 += 1
        print("counter1:", counter1)
        # acc = accuracy_score(y_test, DEC_predict)
        # print('LF-ATTACK - DT - Accuracy of decision tree: ' + str(acc))
        # return acc
    elif model_type == "LR":
        model_bd = myLR.fit(X_train_copy, Y_train_copy)
        LR_predict = myLR.predict(X_test)
        counter1 = 0;
        for i in LR_predict:
            if i == 1:
                counter1 += 1
        print("counter1:", counter1)
        # acc = accuracy_score(y_test, LR_predict)
        # print('LF-ATTACK - LR - Accuracy of logistic regression: ' + str(acc))
        # return acc
    else:
        model_bd = mySVC.fit(X_train_copy, Y_train_copy)
        SVC_predict = mySVC.predict(X_test)
        counter1 = 0;
        for i in SVC_predict:
            if i == 1:
                counter1 += 1
        print("counter1:", counter1)
        # acc = accuracy_score(y_test, SVC_predict)
        # print('LF-ATTACK - SVC - Accuracy of SVC: ' + str(acc))
        # return acc
    return counter1 / len(X_test)


###############################################################################
############################## Evasion ########################################
###############################################################################

def evade_model(trained_model, actual_example):
    # TODO: You need to implement this function!
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    # while pred_class == actual_class:
    # do something to modify the instance
    #    print("do something")

    # eps = 0.0000001;
    # perturb = 0.1
    # threshold = 1
    # threshold_enh= 0.5
    # curr_pert = 0
    # pred_class = actual_class
    # success= 0
    # while pred_class == actual_class:
    #     while curr_pert<threshold:
    #         for i in range(0,3):
    #             modified_example[i] = modified_example[i] + eps
    #             pred_class = trained_model.predict([modified_example])[0]
    #             if(pred_class != actual_class):
    #                 success = 1
    #                 break
    #             per=calc_perturbation(actual_example, modified_example)
    #             print("Perturbation:", per)
    #             modified_example = copy.deepcopy(actual_example)
    #         eps *= 2
    #     if success == 0:
    #         threshold += threshold_enh

    pred_class = actual_class
    a = -0.5
    b = 0.5
    eps = 0.001
    counter = 0
    while pred_class == actual_class:
        mult1 = random.uniform(a, b)
        mult2 = random.uniform(a, b)
        mult3 = random.uniform(a, b)
        mult4 = random.uniform(a, b)
        modified_example[0] *= mult1
        predd = trained_model.predict([modified_example])[0]
        if predd != actual_class:
            return modified_example
        modified_example[1] *= mult2
        predd = trained_model.predict([modified_example])[0]
        if predd != actual_class:
            return modified_example
        modified_example[2] *= mult3
        predd = trained_model.predict([modified_example])[0]
        if predd != actual_class:
            return modified_example
        modified_example[3] *= mult4
        pred_class = trained_model.predict([modified_example])[0]
        a -= eps
        b += eps
        counter += 1
        found = False

        # if pred_class != actual_class:
            # prev = copy.deepcopy(modified_example)
            # while True:
            #     if mult1 > 0:
            #         cleaning1 = -11
            #     else:
            #         cleaning1 = 11
            #     if mult2 > 0:
            #         cleaning2 = -11
            #     else:
            #         cleaning2 = 11
            #     if mult3 > 0:
            #         cleaning3 = -11
            #     else:
            #         cleaning3 = 11
            #     if mult4 > 0:
            #         cleaning4 = -11
            #     else:
            #         cleaning4 = 11
            #
            #     modified_example[0] += cleaning1
            #     modified_example[1] += cleaning2
            #     modified_example[2] += cleaning3
            #     modified_example[3] += cleaning4
            #     last_pred = trained_model.predict([modified_example])[0]
            #     if last_pred == pred_class:
            #         prev = copy.deepcopy(modified_example)
            #     else:
            #         return prev

        # prev = copy.deepcopy(modified_example)
        # if pred_class != actual_class:
        #     pred_tmp = pred_class
        #     prev = copy.deepcopy(modified_example)
        #     modified_example[0] /= mult1
        #     modified_example[1] /= mult2
        #     modified_example[2] /= mult3
        #     modified_example[3] /= mult4
        #     while True:
        #         mult1 *= 0.5
        #         mult2 *= 0.5
        #         mult3 *= 0.5
        #         mult4 *= 0.5
        #         modified_example[0] *= mult1
        #         modified_example[1] *= mult2
        #         modified_example[2] *= mult3
        #         modified_example[3] *= mult4
        #         last_pred = trained_model.predict([modified_example])[0]
        #         if last_pred == pred_class:
        #             prev = copy.deepcopy(modified_example)
        #         else:
        #             return prev

        # prev = copy.deepcopy(modified_example)
        # if pred_class != actual_class:
        #     modified_example[0] /= mult1
        #     modified_example[1] /= mult2
        #     modified_example[2] /= mult3
        #     modified_example[3] /= mult4
        #
        #     last_pred = pred_class
        #     dec = 0.1
        #     k = 0.9
        #     while last_pred == pred_class:
        #         mult1 *= k
        #         mult2 *= k
        #         mult3 *= k
        #         mult4 *= k
        #         modified_example[0] -= mult1
        #         modified_example[1] -= mult2
        #         modified_example[2] -= mult3
        #         modified_example[3] -= mult4
        #         last_pred = trained_model.predict([modified_example])[0]
        #         k -= dec
        #         if last_pred == pred_class:
        #             prev = copy.deepcopy(modified_example)
        #         if last_pred != pred_class or k == 0.5:
        #             return prev

        if counter == 1 and pred_class == actual_class:
            counter = 0
            modified_example = copy.deepcopy(actual_example)

    # modified_example[0] = -2.0
    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!

    actuals = []
    modified_set = []
    for i in range(len(actual_examples)):
        sample = actual_examples[i]
        modified_version = evade_model(DTmodel, sample)
        modified_set.append(modified_version)

    actuals = DTmodel.predict(modified_set)
    # print(actuals)
    lr_classes = LRmodel.predict(modified_set)
    svc_classes = SVCmodel.predict(modified_set)

    lr_success = 0
    svc_success = 0

    for i in range(len(actuals)):
        act_class = actuals[i]
        lr_class = lr_classes[i]
        svc_class = svc_classes[i]
        if act_class == lr_class:
            lr_success +=1
        if act_class == svc_class:
            svc_success += 1

    print("LR Transferability of DT: ", lr_success/len(actual_examples))
    print("SVC Transferability of DT: ", svc_success/len(actual_examples))
    print("-----------------")


    actuals = []
    modified_set = []
    for i in range(len(actual_examples)):
        sample = actual_examples[i]
        modified_version = evade_model(LRmodel, sample)
        modified_set.append(modified_version)

    actuals = LRmodel.predict(modified_set)
    # print(actuals)
    dt_classes = DTmodel.predict(modified_set)
    svc_classes = SVCmodel.predict(modified_set)

    dt_success = 0
    svc_success = 0

    for i in range(len(actuals)):
        act_class = actuals[i]
        dt_class = dt_classes[i]
        svc_class = svc_classes[i]
        if act_class == dt_class:
            dt_success +=1
        if act_class == svc_class:
            svc_success += 1

    print("DT Transferability of LR: ", dt_success/len(actual_examples))
    print("SVC Transferability of LR: ", svc_success/len(actual_examples))
    print("-----------------")

    actuals = []
    modified_set = []
    for i in range(len(actual_examples)):
        sample = actual_examples[i]
        modified_version = evade_model(DTmodel, sample)
        modified_set.append(modified_version)

    actuals = SVCmodel.predict(modified_set)
    # print(actuals)
    lr_classes = LRmodel.predict(modified_set)
    dt_classes = DTmodel.predict(modified_set)

    lr_success = 0
    dt_success = 0

    for i in range(len(actuals)):
        act_class = actuals[i]
        lr_class = lr_classes[i]
        dt_class = dt_classes[i]
        if act_class == lr_class:
            lr_success +=1
        if act_class == dt_class:
            dt_success += 1

    print("LR Transferability of SVC: ", lr_success/len(actual_examples))
    print("DT Transferability of SVC: ", dt_success/len(actual_examples))
    print("-----------------")


    # print(lr_classes)
    # print(svc_classes)
    print("Here, you need to conduct some experiments related to transferability and print their results...")


###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack

    if model_type == "DT":
        actual_labels = remote_model.predict(examples)
        stolen_model = DecisionTreeClassifier(max_depth=5, random_state=0)
        stolen_model.fit(examples, actual_labels)
    elif model_type == "LR":
        actual_labels = remote_model.predict(examples)
        stolen_model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
        stolen_model.fit(examples, actual_labels)
    else:
        actual_labels = remote_model.predict(examples)
        stolen_model = SVC(C=0.5, kernel='poly', random_state=0)
        stolen_model.fit(examples, actual_labels)


    return stolen_model


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    print("LEEENNNNN", len(X_train))

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10, 500]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 50
    total_perturb = 0.0
    for trained_model in trained_models:
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
    print("Avg perturbation for evasion attack:", total_perturb / num_examples)

    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 100
    evaluate_transferability(myDEC, myLR, mySVC, X_test[num_examples:num_examples * 2])

    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()
