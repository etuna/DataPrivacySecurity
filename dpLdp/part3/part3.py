import math, random
import numpy as np

""" Globals """
DOMAIN = list(range(25)) # [0, 1, ..., 24]

""" Helpers """

def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result

# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    """
        Perturbs the given value using GRR protocol.

        Args:
            val (int): User's true value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed value that the user reports to the server
    """
    d = len(DOMAIN)

    p = math.exp(epsilon) / (math.exp(epsilon) + d - 1)
    rand = random.random()

    if rand < p:
        returned_val = val
    else:
        tmp = [item for item in DOMAIN if item != val]
        returned_val = random.choice(tmp)
    # print("val, returnval",val,returned_val)
    # print("p, rand",p,rand)

    return returned_val



# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    """
        Estimates the histogram given GRR perturbed values of the users.

        Args:
            perturbed_values (list): Perturbed values of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    d = len(DOMAIN)
    n = len(perturbed_values)

    p = math.exp(epsilon) / (math.exp(epsilon) + d - 1)
    q = (1-p)/(d-1)

    perturbed_histogram = [0 for x in range(25)]
    for index in range(len(perturbed_values)):
            perturbed_histogram[perturbed_values[index]] += 1

    estimated_histogram = [0 for x in range(25)]
    for index in range(len(perturbed_histogram)):
        nv = perturbed_histogram[index]
        estimated_val = nv*p + (n-nv)*q
        estimated_histogram[index] = estimated_val

    return estimated_histogram

# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for GRR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """
    # error : next(abs(x1 - x2) for (x1, x2) in zip(noisy_hist(estimated), actual_hist(actual))) / len(actual_hist)

    #ACTUAL
    actual_histogram = [0 for x in range(25)]
    for index in range(len(dataset)):
            actual_histogram[dataset[index]] += 1

    # print("ACTUAL:", actual_histogram)

    #PERTURBED
    perturbed_dataset = []
    for data in dataset:
        perturbed_dataset.append(perturb_grr(data, epsilon))

    perturbed_histogram = [0 for x in range(25)]
    for index in range(len(perturbed_dataset)):
            perturbed_histogram[perturbed_dataset[index]] += 1

    # print("PERTURBED:",perturbed_histogram)

    #ESTIMATED
    estimated_histogram = estimate_grr(perturbed_dataset, epsilon)

    # print("ESTIMATED:",estimated_histogram)

    error = next(abs(x1 - x2) for (x1, x2) in zip(estimated_histogram, actual_histogram)) / len(actual_histogram)
    # print(error)
    return error

# TODO: Implement this function!
def encode_rappor(val):
    """
        Encodes the given value into a bit vector.

        Args:
            val (int): The user's true value.
        Returns:
            The encoded bit vector as a list: [0, 1, ..., 0]
    """
    encoded_bitvector = [0 for x in range(25)]
    encoded_bitvector[val] = 1

    return encoded_bitvector

# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    """
        Perturbs the given bit vector using RAPPOR protocol.

        Args:
            encoded_val (list) : User's encoded value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed bit vector that the user reports to the server as a list: [1, 1, ..., 0]
    """
    perturbed_vector = [0 for x in range(25)]
    p = (math.exp(epsilon/2))/(math.exp(epsilon/2)+1)
    q = 1 / (math.exp(epsilon/2)+1)
    choices = ['preserve', 'flip']
    probs = [p, q]
    for index in range(len(encoded_val)):
        bit = encoded_val[index]
        choice = np.random.choice(choices, 1, p=probs)
        if choice == 'flip':
            perturbed_vector[index] = flipBit(bit)
        else:
            perturbed_vector[index] = bit

    return perturbed_vector

def flipBit(bit):
    if bit == 1:
        return 0
    else:
        return 1

# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    """
        Estimates the histogram given RAPPOR perturbed values of the users.

        Args:
            perturbed_values (list of lists): Perturbed bit vectors of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    p = (math.exp(epsilon / 2)) / (math.exp(epsilon / 2) + 1)
    q = 1 / (math.exp(epsilon / 2) + 1)
    n = len(perturbed_values)
    perturbed_histogram = [0 for x in range(25)]
    for perturbed_val in perturbed_values:
        for index in range(len(perturbed_val)):
            bit = perturbed_val[index]
            if bit == 1:
                perturbed_histogram[index] += 1
    estimated_histogram = [0 for x in range(25)]

    for index in range(len(perturbed_histogram)):
        iv = perturbed_histogram[index]
        estimated_val = (iv - n*q) / (p - q)
        estimated_histogram[index] = estimated_val

    return estimated_histogram

    
# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for RAPPOR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """
    #ACTUAL
    actual_histogram = [0 for x in range(25)]
    for data in dataset:
        actual_histogram[data] += 1

    # print("ACTUAL:",actual_histogram)

    # PERTURBED
    perturbed_values = []
    for data in dataset:
        perturbed_val = perturb_rappor(encode_rappor(data), epsilon)
        perturbed_values.append(perturbed_val)
        # print(perturbed_values)

    estimated_histogram = estimate_rappor(perturbed_values, epsilon)
    # print(estimated_histogram)

    error = next(abs(x1 - x2) for (x1, x2) in zip(estimated_histogram, actual_histogram)) / len(actual_histogram)
    # print(error)
    return error

def main():
    dataset = read_dataset("daily_time.txt")

    print(dataset)
    grr_experiment(dataset, 20)

    print("GRR EXPERIMENT")
    #for epsilon in [20.0]:
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    # rappor_experiment(dataset, 2)
    

if __name__ == "__main__":
    main()