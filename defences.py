import numpy as np
from collections import defaultdict
import torch
import pdb
import csv

class DefenseTypes:
    NoDefense = 'NoDefense'
    Krum = 'Krum'
    TrimmedMean = 'TrimmedMean'
    Bulyan = 'Bulyan'
    MedianMean = 'MedianMean'
    MedianMedian = 'MedianMedian'
    MedianMeanKmed = 'MedianMeanKmed'
    MedianMeanRange = 'MedianMeanRange'
    MedianMeanK = 'MedianMeanK'
    MedianMeanNumber = 'MedianMeanNumber'
    MedianMeanNEUP = 'MedianMeanNEUP'
    


    def __str__(self):
        return self.value

def no_defense(users_grads, users_count, corrupted_count, group_size = 3, rate = 10, attack_std = 0.2):
    # pdb.set_trace()
    return torch.mean(users_grads, axis=0)

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances

def krum(users_grads, users_count, corrupted_count, group_size = 3,distances=None,return_index=False, debug=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]


# def median_mean(users_grads, users_count, corrupted_count, group_size = 3):
#     # pdb.set_trace()
#     number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
#     # number_to_consider = 5
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)
    
#     for i, param_across_users in enumerate(users_grads.T):
#         # print("i is", i)
#         if i == 0: # import pdb;pdb.set_trace()
#             mean_vector = torch.zeros(group_size).to(device)
#             num_each_group = int(len(param_across_users) / group_size)
#             if num_each_group * group_size == len(param_across_users):
#                 for ii in range(group_size):
#                     mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
#             else:
#                 for ii in range(group_size-1):
#                     mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
#                 mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])

#             med = torch.median(mean_vector)
            
            
#             c = (param_across_users - med).abs() < 0.3 * abs(med) 
#             # pdb.set_trace()
#             indices = c.nonzero()
#             break

#     select_users_grads = torch.mean(users_grads[indices,:],dim =0)

#     return select_users_grads.view(-1).to(device)


# this is a very good version

def median_mean_NEUP(users_grads, users_count, corrupted_count, group_size = 3, rate = 10): 
    # pdb.set_trace()
    number_to_consider = int(users_grads.shape[0] *0.6) - 1
    # number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
    # number_to_consider = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)

    i = torch.randint(0, users_count,(1,))
    # for i, param_across_users in enumerate(users_grads.T):
    param_across_users = users_grads.T[i][0].view(-1)
    # for i, param_across_users in enumerate(users_grads.T):
        # print("i is", i)
        # if i == 0: # import pdb;pdb.set_trace()
    mean_vector = torch.zeros(group_size).to(device)
    num_each_group = int(len(param_across_users) / group_size)
    if num_each_group * group_size == len(param_across_users):
        for ii in range(group_size):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
    else:
        for ii in range(group_size-1):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
        mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])


    med = torch.median(mean_vector)
    c = torch.abs(param_across_users - med) < torch.abs(rate* med)
    # pdb.set_trace()
    # c = (param_across_users - med).abs() < (0.1* med)
    indices = c.nonzero()
    print("indices number is", len(indices))
        # break
    good_vals  = param_across_users[indices]
    # current_grads[i] = torch.mean(good_vals)
    select_users_grads = torch.mean(users_grads[indices,:],dim =0)
    # pdb.set_trace()
        
        # current_grads[i] = np.median(mean_vector)
    return select_users_grads.view(-1).to(device)



def median_mean_range(users_grads, users_count, corrupted_count, group_size = 3, rate = 10, attack_std = 0.2): 
    # pdb.set_trace()
    number_to_consider = int(users_grads.shape[0] *0.6) - 1
    # number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
    # number_to_consider = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pdb.set_trace()
    current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)
    # tem_indices = []
    for sample_times in range(int(users_grads.shape[1]*0.01)):
        i = torch.randint(0, users_count,(1,))
        # for i, param_across_users in enumerate(users_grads.T):
        param_across_users = users_grads.T[i][0]
        mean_vector = torch.zeros(group_size).to(device)
        num_each_group = int(len(param_across_users) / group_size)
        if num_each_group * group_size == len(param_across_users):
            for ii in range(group_size):
                mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
        else:
            for ii in range(group_size-1):
                mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
            mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])


        med = torch.median(mean_vector)
        # cluster_mean = torch.mean(mean_vector)
        small_indices = torch.argsort(torch.abs(mean_vector - med))[:2]
        threshold = torch.abs(mean_vector[small_indices[1]] - med)
        # smallest_value =  torch.abs(mean_vector - med).min()
        c = torch.abs(param_across_users - med) <= torch.abs(rate * threshold)

        # pdb.set_trace()
        # open the file in the write mode
        if sample_times == 0:
            with open('./logs/Krange_CIFAR10_attack_'+str(attack_std)+'_'+str(rate)+'.csv', "a") as fp:
                fp.write(str(abs((rate * threshold).cpu().numpy())))
                fp.write('\n')
        # f = open('./logs/Krange_'+str(rate)+'.csv', 'w')
        # # create the csv writer
        # writer = csv.writer(f)
        # # write a row to the csv file
        # writer.writerow(torch.abs(rate * threshold).cpu().numpy())
        # # close the file
        # f.close()
        indices = c.nonzero()

        
        if sample_times == 0:
            merged_indices = indices.view(-1)
        else:
            merged_indices = torch.cat((indices.view(-1), merged_indices), 0)
        # smallest_value =  torch.abs(param_across_users - cluster_mean).min()
        # indices = torch.argsort(torch.abs(param_across_users - cluster_mean))[:number_to_consider]
    ferquency_indices = torch.bincount(merged_indices)
    # print("ferquency_indices is", ferquency_indices)
    # pdb.set_trace()

    d = ferquency_indices > 1
    final_indices = d.nonzero()
    
    include_malicious = final_indices < corrupted_count
    include_malicious_count = include_malicious.nonzero()
    log_filepath = './log_malicious'
    
    with open(log_filepath, 'a') as fh:
        fh.write("group_size:{}, k:{}, attack_std:{}, include_malicious_count:{},total_user:{},\n".format(group_size, rate, attack_std,include_malicious_count.shape[0], users_count))
        
    # pdb.set_trace() 
    # c = torch.abs(param_across_users - cluster_mean) <= torch.abs(rate * smallest_value)
    # indices = c.nonzero()
    # print("indices number is", len(indices))
    # pdb.set_trace()
    # good_vals  = param_across_users[indices]
    # current_grads[i] = torch.mean(good_vals)
    select_users_grads = torch.mean(users_grads[final_indices,:],dim =0)
    
    return select_users_grads.view(-1).to(device)
    # pdb.set_trace()

def median_mean_k_med(users_grads, users_count, corrupted_count, group_size = 3, rate = 10): 
    # pdb.set_trace()
    number_to_consider = int(users_grads.shape[0] *0.6) - 1
    # number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
    # number_to_consider = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)

    i = torch.randint(0, users_count,(1,))
    # for i, param_across_users in enumerate(users_grads.T):
    param_across_users = users_grads.T[i][0].view(-1)
    # for i, param_across_users in enumerate(users_grads.T):
        # print("i is", i)
        # if i == 0: # import pdb;pdb.set_trace()
    mean_vector = torch.zeros(group_size).to(device)
    num_each_group = int(len(param_across_users) / group_size)
    if num_each_group * group_size == len(param_across_users):
        for ii in range(group_size):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
    else:
        for ii in range(group_size-1):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
        mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])


    med = torch.median(mean_vector)
    c = torch.abs(param_across_users - med) < torch.abs(rate* med)
    # pdb.set_trace()
    # c = (param_across_users - med).abs() < (0.1* med)
    indices = c.nonzero()
    print("indices number is", len(indices))
        # break
    good_vals  = param_across_users[indices]
    # current_grads[i] = torch.mean(good_vals)
    select_users_grads = torch.mean(users_grads[indices,:],dim =0)
    # pdb.set_trace()
        
        # current_grads[i] = np.median(mean_vector)
    return select_users_grads.view(-1).to(device)



def median_mean_number(users_grads, users_count, corrupted_count, group_size = 3, rate = 10): 
    # pdb.set_trace()
    number_to_consider = int(users_grads.shape[0] *0.6) - 1
    # number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
    # number_to_consider = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)
    # pdb.set_trace()
    i = torch.randint(0, users_count,(1,))
    # for i, param_across_users in enumerate(users_grads.T):
    param_across_users = users_grads.T[i].view(-1)
    mean_vector = torch.zeros(group_size).to(device)
    num_each_group = int(len(param_across_users) / group_size)
    if num_each_group * group_size == len(param_across_users):
        for ii in range(group_size):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
    else:
        for ii in range(group_size-1):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
        mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])

    med = torch.median(mean_vector)
    # pdb.set_trace()
    indices = torch.argsort(torch.abs(param_across_users - med))[:number_to_consider]

    select_users_grads = torch.mean(users_grads[indices,:],dim =0)
    # pdb.set_trace()
        
        # current_grads[i] = np.median(mean_vector)
    return select_users_grads.view(-1).to(device)
    # return current_grads.to(device)


def median_mean_k(users_grads, users_count, corrupted_count, group_size = 3, rate = 10): 
    # pdb.set_trace()
    number_to_consider = int(users_grads.shape[0] *0.6) - 1
    # number_to_consider = int((users_grads.shape[0] - corrupted_count)*0.6) - 1
    # number_to_consider = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_grads = torch.zeros(users_grads.shape[1], dtype = users_grads.dtype).to(device)
    
    i = torch.randint(0, users_count,(1,))
    # for i, param_across_users in enumerate(users_grads.T):
    param_across_users = users_grads.T[i][0].view(-1)
    # for i, param_across_users in enumerate(users_grads.T):
        # print("i is", i)
        # if i == 0: # import pdb;pdb.set_trace()
    mean_vector = torch.zeros(group_size).to(device)
    num_each_group = int(len(param_across_users) / group_size)
    if num_each_group * group_size == len(param_across_users):
        for ii in range(group_size):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
    else:
        for ii in range(group_size-1):
            mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
        mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])


    med = torch.median(mean_vector)
    # c = torch.abs(param_across_users - med) < torch.abs(50* med)
    # # pdb.set_trace()
    # # c = (param_across_users - med).abs() < (0.1* med)
    # indices = c.nonzero()
    # print("indices number is", len(indices))
    #     # break
    # good_vals  = param_across_users[indices]
    # # current_grads[i] = torch.mean(good_vals)
    # select_users_grads = torch.mean(users_grads[indices,:],dim =0)
    # # pdb.set_trace()
    x = torch.abs(param_across_users - med)**2
    std = torch.sqrt(torch.mean(x))
    c = (param_across_users - med).abs() < 1*std * rate/100
    indices = c.nonzero()
            # break
        #     good_vals  = param_across_users[indices]
        #     current_grads[i] = torch.mean(good_vals)
        # else:
        #     good_vals  = param_across_users[indices]
        #     current_grads[i] = torch.mean(good_vals)
    # pdb.set_trace()
    select_users_grads = torch.mean(users_grads[indices,:],dim =0)


    return select_users_grads.view(-1).to(device)
    # return current_grads.to(device)


def median_median(users_grads, users_count, corrupted_count, group_size = 3, rate = 10):
    number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)
    group = group_size
    for i, param_across_users in enumerate(users_grads.T):
        # import pdb;pdb.set_trace()
        
        mean_vector = np.zeros(group)
        num_each_group = int(len(param_across_users) / group)
        for ii in range(group):
            mean_vector[ii] = np.median(param_across_users[ii*num_each_group:(ii+1)*num_each_group-1])

        # med = np.median(param_across_users)
        # good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        # current_grads[i] = np.mean(good_vals) + med
        current_grads[i] = np.median(mean_vector)
    return current_grads

def trimmed_mean(users_grads, users_count, corrupted_count, group_size = 3, rate = 10):
    number_to_consider = int(users_grads.shape[0] - corrupted_count) - 1
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)

    for i, param_across_users in enumerate(users_grads.T):
        # import pdb;pdb.set_trace()
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_grads[i] = np.mean(good_vals) + med 
    return current_grads


def bulyan(users_grads, users_count, corrupted_count, rate = 10):
    assert users_count >= 4*corrupted_count + 3
    set_size = users_count - 2*corrupted_count
    selection_set = []

    distances = _krum_create_distances(users_grads)
    while len(selection_set) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, distances, True)
        selection_set.append(users_grads[currently_selected])

        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)

    return trimmed_mean(np.array(selection_set), len(selection_set), 2*corrupted_count)


defend = {DefenseTypes.Krum: krum,
          DefenseTypes.TrimmedMean: trimmed_mean, 
          DefenseTypes.MedianMeanNEUP: median_mean_NEUP,
          DefenseTypes.MedianMeanNumber: median_mean_number,
          DefenseTypes.MedianMeanRange: median_mean_range,
          DefenseTypes.MedianMedian: median_median, 
          DefenseTypes.MedianMeanKmed: median_mean_k_med,
          DefenseTypes.MedianMeanK: median_mean_k,
          DefenseTypes.NoDefense: no_defense,
          DefenseTypes.Bulyan: bulyan}