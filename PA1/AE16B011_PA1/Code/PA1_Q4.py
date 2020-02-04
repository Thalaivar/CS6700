import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import statistics

N_BANDITS = 10

np.random.seed(124)

normal = np.random.normal
cmap = mpl.cm.get_cmap('tab20')

def MedianElimination(eps, delta):
	n_arms = N_BANDITS

	# true rewards
	mu = normal(0, 1, (n_arms,))

	# print("True best action = %d" % (np.argmax(mu)))
	var = np.ones((n_arms,))
	eps = eps/4
	delta = delta/2
	
	# expected rewards for arms
	Q_a_est = np.zeros((n_arms,))
	# initial action set consists of all arms
	arms = np.ones((n_arms,))
	
	total_plays = 0
	n_t = 0
	while(n_arms > 1):
		N_PLAYS = int((1/(eps*0.5)**2)*np.log(3/delta))
		total_plays += N_PLAYS
		n_t += total_plays*n_arms
		
		# sample all arms
		for i in range(N_PLAYS):
			reward = normal(mu, var)
			Q_a_est += reward

		p_a = Q_a_est/total_plays
		ml = findMedian(p_a, arms)
		
		# reduce action set
		for i in range(N_BANDITS):
			if p_a[i] < ml and arms[i] == 1:
				arms[i] = 0

		n_arms = int(np.sum(arms))

		eps = 0.75*eps
		delta = delta*0.5

		# print("|A| = %d" % (n_arms))

	best_action = np.argmax(arms)
	return np.argmax(mu), best_action, n_t

def findMedian(Q_a_est, arms):
	n_arms = int(np.sum(arms))
	new_Q_a = np.zeros((n_arms,))
	j = 0
	for i in range(np.size(arms)):
		if arms[i] == 1:
			new_Q_a[j] = Q_a_est[i];
			j += 1

	return statistics.median(new_Q_a)

def basic_run():
	eps = 10
	delta = 0.1

	true_best_action, best_action, total_plays = MedianElimination(eps, delta)
	print("True best: %d ; Final action: %d ; Total plays = %d" % (true_best_action, best_action, total_plays))

def explore_eps():
	eps = np.linspace(0.5, 10, 100)
	delta = 0.1

	plays = np.zeros((len(eps),))
	for i in range(len(eps)):
		true_best, best, total_plays = MedianElimination(eps[i], delta)
		if true_best == best:
			plays[i] = total_plays

	plt.plot(eps, plays)
	plt.show()

def main():
	# basic_run()
	explore_eps()

if __name__ == '__main__':
	main()