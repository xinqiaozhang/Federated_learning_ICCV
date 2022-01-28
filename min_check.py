def choose_k(N, k):
    # choose k objects out of N
    if k > N:
        return 0
    count = 1
    for i in range(k):
        count *= (N-i)/(i+1)
    return count
def hypergeom(N, K, k, n):
    # probability of k successes in n draws, without replacement,
    # from a population size of N that contains K objects with that feature
    count_success = choose_k(K, k) * choose_k(N-K, n-k)
    count_total = choose_k(N, n)
    return count_success/count_total
def prob_success(N, K, n):
    return 1. - hypergeom(N, K, k=0, n=n)

N = 44426 # number of parameters
p = 0.33 # percentage of parameters with noise
failure_prob = 0.01
K = int(p*N)
range_n = range(1, N-K+2)
for n in range_n:
    p_success = prob_success(N, K, n)
    if 1-p_success < failure_prob:
        min_checks = n
        break
print(min_checks)