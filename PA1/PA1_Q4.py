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

	# true_best_action, best_action, total_plays = MedianElimination(eps, delta)
	# print("True best: %d ; Final action: %d ; Total plays = %d" % (true_best_action, best_action, total_plays))

	plays = 0
	success = 0
	for i in range(2000):
		true_best_action, best_action, total_plays = MedianElimination(eps, delta)
		if true_best_action == best_action:
			success += 1
			plays += total_plays

	print("Success: %d ; Average plays taken: %d" % (success, plays/success))

def explore_eps():
	eps = np.linspace(0.1, 10, 10)
	delta = 0.1

	avg_plays = np.zeros((len(eps),))
	success = np.zeros((len(eps),))
	for i in range(len(eps)):
		for j in range(100):
			true_best_action, best_action, total_plays = MedianElimination(eps[i], delta)
			if true_best_action == best_action:
				success[i] += 1
				avg_plays[i] += total_plays

		avg_plays[i] = avg_plays[i]/success[i]

		print(i/len(eps))

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	plt.plot(eps, np.log(avg_plays), color=cmap(0.4))
	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()  
	plt.yticks(fontsize=12)
	plt.xticks(fontsize=12)
	plt.xlabel(r'$\epsilon$', fontsize=14)
	plt.ylabel(r'$\log(\mathrm{plays})$', fontsize=14)
	plt.title(r'Average plays performed vs. $\epsilon$')
	plt.show()

	plt.figure(figsize=(12, 9))
	ax = plt.subplot(111)
	plt.plot(eps, success/2000, color=cmap(0.4))
	ax.get_xaxis().tick_bottom()    
	ax.get_yaxis().tick_left()  
	plt.yticks(fontsize=12)
	plt.xticks(fontsize=12)
	plt.xlabel(r'$\epsilon$', fontsize=14)
	plt.ylabel(r'$\%$ of successful runs', fontsize=14)
	plt.title(r'Successful runs vs. $\epsilon$')
	plt.show()

def main():
	# basic_run()

	# takes a fair bit of time
	explore_eps()

if __name__ == '__main__':
	main()