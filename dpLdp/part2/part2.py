import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
from numpy import real
from pandas.plotting._matplotlib import hist

''' Globals '''
LABELS = ["", "frontpage", "news", "tech", "local", "opinion", "on-air", "misc", "weather", "msn-news", "health", "living" , "business", "msn-sports", "sports", "summary", "bbs", "travel"]


""" 
    Helper functions
    (You can define your helper functions here.)
"""
def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list of lists.
    """

    result = []
    with open(filename, "r") as f:
        for _ in range(7):
            next(f)
        for line in f:
            sequence = line.strip().split(" ")
            result.append([int(i) for i in sequence])
    return result


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset: list, path='np-histogram.png'):
    """
        Creates a histogram of given counts for each category and saves it to a file.

        Args:
            dataset (list of lists): The MSNBC dataset

        Returns:
            Ordered list of counts for each page category (frontpage, news, tech, ..., travel)
            Ex: [123, 383, 541, ..., 915]
    """
    histogram = [0 for x in range(18)]
    for index in range(len(dataset)):
        for i in range(len(dataset[index])):
            histogram[dataset[index][i]] += 1


    df = pd.DataFrame(columns=LABELS)
    df.loc[0] = histogram

    data = df.iloc[0].to_dict()
    cats = list(data.keys())
    vals = list(data.values())
    y_pos = np.arange(len(cats))

    plt.bar(y_pos[1:], vals[1:])
    plt.title('# of visits in each category')
    plt.xlabel('categories')
    plt.xticks(y_pos, cats, rotation=90, fontsize=8)
    plt.ylabel('Counts', fontsize=8)
    # plt.show()
    plt.savefig(path, bbox_inches = 'tight')
    # print(vals[1:])
    return vals[1:]

# TODO: Implement this function!
def add_laplace_noise(real_answer: list, sensitivity: float, epsilon: float):
    """
        Adds laplace noise to query's real answers.

        Args:
            real_answer (list): Real answers of a query -> Ex: [92.85, 57.63, 12, ..., 15.40]
            sensitivity (float): Sensitivity
            epsilon (float): Privacy parameter
        Returns:
            Noisy answers as a list.
            Ex: [103.6, 61.8, 17.0, ..., 19.62]
    """

    # print(real_answer)

    # 1) Compute the real answer of query: q(D)
    # 2) Find sensitivity of q: S(q)
    # 3) Draw a random sample from Lap(0, S(q)/𝛆)
    # ▪ Or Lap(S(q)/𝛆) for short
    # 4) Add the random sample r to real answer q(D):
    # q'(D) = q(D) + r, where q'(D) becomes the noisy
    # answer
    # 5) Return q'(D) to the user

    laplace_noise = [np.random.laplace(loc=0, scale=sensitivity / epsilon) for x in range(len(real_answer))]
    noisy_answer = [x1 + x2 for(x1, x2) in zip(real_answer, laplace_noise)]
    # print(laplace_noise)
    # print(noisy_answer)
    return noisy_answer

# TODO: Implement this function!
def truncate(dataset: list, n: int):
    """
        Truncates dataset according to truncation parameter n.

        Args:  
            dataset: original dataset 
            n (int): truncation parameter
        Returns:
            truncated_dataset: truncated version of original dataset
    """
    truncated_dataset = []
    for index in range(len(dataset)):
        truncated_sequence = []
        for i in range(len(dataset[index])):
            if i < n:
                truncated_sequence.append(dataset[index][i])
        truncated_dataset.append(truncated_sequence)
    # print(dataset[:20])
    # print(truncated_dataset[:20])
    return truncated_dataset


# TODO: Implement this function!
def get_dp_histogram(dataset: list, n: int, epsilon: float):
    """
        Truncates dataset with parameter n and calculates differentially private histogram.

        Args:
            dataset (list of lists): The MSNBC dataset
            n (int): Truncation parameter
            epsilon (float): Privacy parameter
        Returns:
            Differentially private histogram as a list
    """
    savePath = 'dp_histogram.png'
    truncated_dataset = truncate(dataset, n)
    dph_data = get_histogram(truncated_dataset, savePath)
    noisy_dataset = add_laplace_noise(dph_data, 1, epsilon)
    # noisy_dataset = add_laplace_noise(truncated_dataset,1, epsilon)
    # dph_data = get_histogram(noisy_dataset, 'dp_histogram.png')

    df = pd.DataFrame(columns=LABELS)
    df.loc[0] = [0] + noisy_dataset

    data = df.iloc[0].to_dict()
    cats = list(data.keys())
    vals = list(data.values())
    y_pos = np.arange(len(cats))

    plt.figure(2)
    plt.bar(y_pos[1:], vals[1:])
    plt.title('# of visits in each category[DP]')
    plt.xlabel('categories')
    plt.xticks(y_pos, cats, rotation=90, fontsize=8)
    plt.ylabel('Counts', fontsize=8)
    # plt.show()
    plt.savefig(savePath, bbox_inches='tight')


    return noisy_dataset


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    """
        Calculates error according to the equation stated in part (e).

        Args: Actual histogram (list), Noisy histogram (list)
        Returns: Error (Err) in the noisy histogram (float)
    """
    return next(abs(x1 - x2) for (x1, x2) in zip(noisy_hist, actual_hist)) / len(actual_hist)

# TODO: Implement this function!
def n_experiment(dataset, n_values: list, epsilon: float):
    """
        Function for the experiment explained in part (f).
        n_values is a list, such as: [1, 6, 11, 16 ...]
        Returns the errors as a list: [1256.6, 1653.5, ...] such that 1256.5 is the error when n=1,
        1653.5 is the error when n = 6, and so forth.
    """
    n_times = 30
    errors_n30 = [0 for x in range(n_times)]
    for ind in range(n_times):
        errors = [0 for x in range(len(n_values))]
        actual_hist = get_histogram(dataset)
        for i in range(len(n_values)):
            noisy_hist = get_dp_histogram(dataset, n_values[i], epsilon)
            errors[i] = calculate_average_error(actual_hist, noisy_hist)
        errors_n30[ind] = errors
        # print("n_exp, ind:", ind)


    errors_avg = np.mean(errors_n30, axis=0)
    # print(errors_n30)
    # print("AVGGGGGGG:", errors_avg)
    return errors_avg


# TODO: Implement this function!
def epsilon_experiment(dataset, n: int, eps_values: list):
    """
        Function for the experiment explained in part (g).
        eps_values is a list, such as: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
        Returns the errors as a list: [9786.5, 1234.5, ...] such that 9786.5 is the error when eps = 0.0001,
        1234.5 is the error when eps = 0.001, and so forth.
    """
    n_times = n
    errors_n30 = [0 for x in range(n_times)]
    for ind in range(n_times):
        errors = [0 for x in range(len(eps_values))]
        actual_hist = get_histogram(dataset)
        for i in range(len(eps_values)):
            noisy_hist = get_dp_histogram(dataset, n, eps_values[i])
            errors[i] = calculate_average_error(actual_hist, noisy_hist)
        errors_n30[ind] = errors

    errors_avg = np.mean(errors_n30, axis=0)
    # print(errors_n30)
    # print("AVGGGGGGG:", errors_avg)
    return errors_avg


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def extract(dataset):
    """
        Extracts the first 1000 sequences and truncates them to n=1
    """
    return truncate(dataset[:1000], 1)




# TODO: Implement this function!
def most_visited_exponential(smaller_dataset, epsilon):
    """
        Using the Exponential mechanism, calculates private response for query: 
        "Which category (1-17) received the highest number of page visits?"

        Returns 1 for frontpage, 2 for news, 3 for tech, etc.
    """
    # 1) Determine the appropriate score function qF
    # 2) Compute its sensitivity: S(qF)
    # 3) For all r in R, compute qF(D,r) by executing the function on the dataset D.
    # 4)Exponential mechanism picks and returns one r* from R as its final result with probability:

    # qF(D, category) : # of visit of the category
    # sensitivity : 1
    # 3.step is the histogram

    choices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    sensitivity = 1
    probs = [0 for x in range(17)]
    qF_actual = get_histogram(smaller_dataset)
    # print("&"*100)
    # print(qF_actual)
    # print("qf actual len:",str(len(qF_actual)))
    # print("rangelen, epsilon, math.exp", range(len(qF_actual)), epsilon,math.exp(epsilon*qF_actual[3]/2*sensitivity))
    common_divisor = sum(math.exp(epsilon*qF_actual[i]/(2*sensitivity)) for i in range(len(qF_actual)))
    # print(common_divisor)
    for index in range(len(qF_actual)):
        divident = math.exp(epsilon*qF_actual[index]/(2*sensitivity))
        probs[index] = divident / common_divisor
        # print(index,":", divident," ----->", str(divident/common_divisor))

    return np.random.choice(choices, 1, p=probs)


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    """
        Function for the experiment explained in part (i).
        eps_values is a list such as: [0.001, 0.005, 0.01, 0.03, ..]
        Returns the list of accuracy results [0.51, 0.78, ...] where 0.51 is the accuracy when eps = 0.001,
        0.78 is the accuracy when eps = 0.005, and so forth.
    """
    correct_ans = 1
    num_trials = 50
    smaller_dataset = extract(dataset)
    accuracies = [0 for x in range(len(eps_values))]
    for index in range(len(eps_values)):
        for i in range(num_trials):
            returned_val = most_visited_exponential(smaller_dataset, eps_values[index])
            if returned_val == correct_ans:
                accuracies[index] += 1
        accuracies[index] /= num_trials
        # print("eps index:", index)
    return accuracies



# FUNCTIONS TO IMPLEMENT END #

def main():
    dataset_filename = "msnbc.dat"
    dataset = read_dataset(dataset_filename)

    non_private_histogram = get_histogram(dataset)
    print("Non private histogram:", non_private_histogram)

    noisy_histogram =  get_dp_histogram(dataset, 40, 3)
    print("Noisy histogram:", noisy_histogram)

    # # add_laplace_noise([1,2,3,4], 2, 3)
    # error = calculate_average_error(non_private_histogram, noisy_histogram)
    # print(error)
    #
    # most_visited_exponential(extract(dataset), 0.01)

    print("**** N EXPERIMENT RESULTS (f of Part 2) ****")
    eps = 0.01
    n_values = []
    for i in range(1, 106, 5):
       n_values.append(i)
    errors = n_experiment(dataset, n_values, eps)
    for i in range(len(n_values)):
       print("n = ", n_values[i], " error = ", errors[i])

    print("*" * 50)
    #
    print("**** EPSILON EXPERIMENT RESULTS (g of Part 2) ****")
    n = 1000
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    errors = epsilon_experiment(dataset, n, eps_values)
    for i in range(len(eps_values)):
       print("eps = ", eps_values[i], " error = ", errors[i])

    print("*" * 50)

    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
       print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()
