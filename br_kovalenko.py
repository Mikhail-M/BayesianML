# -*- coding: utf-8 -*-
import numpy as np
from math import factorial, floor, sqrt
import matplotlib.pyplot as plt
from scipy.misc import comb
import scipy, scipy.stats
from scipy.stats import binom
np.set_printoptions(threshold=np.nan)

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

# # Homework parametrs
# params = {'amin':75, 'amax':90, 'bmin':500, 'bmax':600, 'p1':0.1, 'p2':0.01, 'p3':0.3}
# Ea = 83
# Eb = 550
# Ed = 18


def pc(params = None, model = 2):
	if model == 2:
		
		a = params['amax'] - params['amin'] +1
		b = params['bmax'] - params['bmin'] +1
		Distr_a = 1.0 / a
		Distr_b = 1.0 / b

		# calculate matrix of lambdas and matrix of exponents of lambdas
		lambdas = np.zeros([a , b])
		exp_lambdas = np.zeros([lambdas.shape[0], lambdas.shape[1]])

		i = 0
		j = 0

		for a_i in range(params['amin'], params['amax'] +1):
			for b_i in range(params['bmin'], params['bmax'] +1):
				lambdas[i,j] = a_i*params['p1'] + b_i*params['p2']
				j += 1
			i += 1
			j = 0

		exp_lambdas = np.exp(-1*lambdas)


		# calculate disctribution for c
		Distr_c = np.zeros(params['amax'] + params['bmax'] + 1)

		for c in range(0, params['amax'] + params['bmax'] + 1):
			summa = 0.0
			for (x,y), v in np.ndenumerate(lambdas):
				summa = summa + v**c * exp_lambdas[x,y]
			if summa != 0.0:
				Distr_c[c] = summa * Distr_a * Distr_b / factorial(c)
			if c > 150:
				break
		return Distr_c, np.arange(Distr_c.shape[0])
	
	if model == 1:
		a = params['amax'] - params['amin'] + 1
		b = params['bmax'] - params['bmin'] + 1
		Distr_a = 1.0 / a
		Distr_b = 1.0 / b
		
		# Precalculate vector of a - sums 
		A_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for a_i in reversed(range(params['amin'], params['amax'] +1)):
				if a_i < c:
					break
				p_a = binom.pmf(c, a_i, params['p1'])
				
				sums = sums + p_a
			A_sums[c] = sums

		# Precalculate vector of b - sums 
		B_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		threshold = 3
		for c in range(0, params['amax'] + params['bmax'] + 1):
			if c > params['bmax']:
				break
			sums = 0.0
			
			for b_i in reversed(range(params['bmin'], params['bmax'] +1)):
				if b_i < c:
					break
				p_b = binom.pmf(c, b_i, params['p2'])
				sums = sums + p_b
			
			if sums < 0.1:
				threshold -= 1
			if threshold == 0:
				break
			B_sums[c] = sums
		
		# calculate distibution for c
		C_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for cs in range(0, c+1):
				if A_sums[cs] == 0:
					continue
				sums = sums + A_sums[cs] * B_sums[c - cs]
			C_sums[c] = sums * Distr_a * Distr_b

		return C_sums, np.arange(C_sums.shape[0])



def pd(params = None, model=2):
	Distr_c, supp = pc(params = params, model = model) 
	# if len(Distr_c) == 0:
	# 	print "Distribution of P(c) is not provided!"
	# 	return False

	if model == 1 or model == 2:
		c_numbers = np.arange(Distr_c.shape[0])
		Distr_c = list(Distr_c)
		c_numbers = list(c_numbers)

		p = params['p3']
		q = 1-p

		# calculate disctribution for d
		Distr_d = np.zeros(2*(params['amax'] + params['bmax']) + 1)
		threshold = 5
		for d in range(0, (2*(params['amax']+params['bmax']) + 1)):
			summa = 0.0
			for (cn, c) in zip(c_numbers, Distr_c):
				if cn > d:
					break
				u = d - cn
				if u > cn:
					continue
				summa = summa + c * comb(cn, u, exact=False) * p**u * q**(cn - u)
			if summa < 0.0001:
				threshold -= 1
			if threshold == 0:
				break
			Distr_d[d] = summa
		
		return Distr_d, np.arange(Distr_d.shape[0])



def pc_a(params = None, a = 83, model = 2):
	Ea = a
	if model == 2:
		b = params['bmax'] - params['bmin'] + 1
		Distr_b = 1.0 / b

		a = Ea
		# calculate matrix of lambdas and matrix of exponents of lambdas
		lambdas = np.zeros(b)
		exp_lambdas = np.zeros(lambdas.shape[0])

		i = 0
		for b_i in range(params['bmin'], params['bmax'] +1):
			lambdas[i] = a*params['p1'] + b_i*params['p2']
			i += 1

		exp_lambdas = np.exp(-1*lambdas)

		# calculate disctribution for c_a
		threshold = 10
		Distr_c = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			summa = 0.0
			for x, v in np.ndenumerate(lambdas):
				summa = summa + v**c * exp_lambdas[x]
			if summa != 0.0:
				Distr_c[c] = summa * Distr_b / factorial(c)
			if c > 160:
				break
		return Distr_c, np.arange(Distr_c.shape[0])

	if model == 1: 
		b = params['bmax'] - params['bmin'] + 1
		Distr_b = 1.0 / b
		
		a = Ea

		# Precalculate vector of a - sums 
		A_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			A_sums[c] = binom.pmf(c, a, params['p1'])

		# Precalculate vector of b - sums 
		B_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		threshold = 5
		for c in range(0, params['amax'] + params['bmax'] + 1):
			if c > params['bmax']:
				break
			sums = 0.0
			
			for b_i in reversed(range(params['bmin'], params['bmax'] +1)):
				if b_i < c:
					break
				p_b = binom.pmf(c, b_i, params['p2'])
				sums = sums + p_b
			
			if sums < 0.1:
				threshold -= 1
			if threshold == 0:
				break
			B_sums[c] = sums
		
		# calculate distibution for c
		C_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for cs in range(0, c+1):
				if A_sums[cs] == 0:
					continue
				sums = sums + A_sums[cs] * B_sums[c - cs]
			C_sums[c] = sums * Distr_b

		return C_sums, np.arange(C_sums.shape[0])



def pc_b(params = None, b = 550, model = 2):
	Eb = b
	if model == 2:
		a = params['amax'] - params['amin'] + 1
		Distr_a = 1.0 / a

		b = Eb
		# calculate matrix of lambdas and matrix of exponents of lambdas
		lambdas = np.zeros(a)
		exp_lambdas = np.zeros(lambdas.shape[0])

		i = 0
		for a_i in range(params['amin'], params['amax'] +1):
			lambdas[i] = a_i*params['p1'] + b*params['p2']
			i += 1
		exp_lambdas = np.exp(-1*lambdas)

		# calculate disctribution for c_b
		Distr_c_b = np.zeros(params['amax'] + params['bmax'] + 1)

		for c in range(0, params['amax'] + params['bmax'] + 1):
			summa = 0.0
			for x, v in np.ndenumerate(lambdas):
				summa = summa + v**c * exp_lambdas[x]
			if summa != 0.0:
				Distr_c_b[c] = summa * Distr_a / factorial(c)
			if c > 150:
				break

		return Distr_c_b, np.arange(Distr_c_b.shape[0])

	if model == 1:
		a = params['amax'] - params['amin'] + 1
		Distr_a = 1.0 / a

		b = Eb
		
		# Precalculate vector of a - sums 
		A_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for a_i in reversed(range(params['amin'], params['amax'] +1)):
				if a_i < c:
					break
				p_a = binom.pmf(c, a_i, params['p1'])
				
				sums = sums + p_a
			A_sums[c] = sums

		# Precalculate vector of b - sums 
		B_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		threshold = 5
		for c in range(0, params['amax'] + params['bmax'] + 1):
			if c > params['bmax']:
				break

			B_sums[c] = binom.pmf(c, b, params['p2'])
		
		# calculate distibution for c
		C_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for cs in range(0, c+1):
				if A_sums[cs] == 0:
					continue
				sums = sums + A_sums[cs] * B_sums[c - cs]
			C_sums[c] = sums * Distr_a

		return C_sums, np.arange(C_sums.shape[0])


def pc_ab(params = None, a = 83, b = 550, model = 2):
	Ea = a
	Eb = b

	if model == 2:
		lambda_for_d = Ea*params['p1'] + Eb*params['p2']

		# calculate disctribution for c_ab
		Distr_c_ab = np.zeros(params['amax'] + params['bmax'] + 1)

		for c in range(0, params['amax'] + params['bmax'] + 1):
			Distr_c_ab[c] = lambda_for_d**c * 2.718281**(-lambda_for_d) / factorial(c)
			if c > 150:
				break
		return Distr_c_ab, np.arange(Distr_c_ab.shape[0])

	if model == 1:

		a = Ea
		b = Eb
		
		# Precalculate vector of a - sums 
		A_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			A_sums[c] = binom.pmf(c, a, params['p1'])

		# Precalculate vector of b - sums 
		B_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		threshold = 5
		for c in range(0, params['amax'] + params['bmax'] + 1):
			if c > params['bmax']:
				break
			B_sums[c] = binom.pmf(c, b, params['p2'])
		
		# calculate distibution for c
		C_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for cs in range(0, c+1):
				if A_sums[cs] == 0:
					continue
				sums = sums + A_sums[cs] * B_sums[c - cs]
			C_sums[c] = sums

		return C_sums, np.arange(C_sums.shape[0])



def pc_d(params = None, d = 18, model = 2):
	Ed = d

	p_c, supp = pc(params = params, model = model)
	p_d, supp = pd(params = params, model = model)

	if model == 2 or model == 1:
		probd = p_d[Ed]
		p = params['p3']
		q = 1-p

		# calculate disctribution for c_ab
		Distr_c_d = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			succes = Ed - c
			if succes < 0:
				break
			Distr_c_d[c] = 1/probd * p_c[c] * comb(c, succes) * p**succes * q**(c - succes)
		return Distr_c_d, np.arange(Distr_c_d.shape[0])


def pc_abd(params = None, model = 2, a = 83, b = 550, d = 18):
	Ea = a
	Eb = b
	Ed = d
	if model == 2:
		lambda_ = Ea * params['p1'] + Eb * params['p2']
		denomin_sum = 0.0
		for c in range(0, params['amax'] + params['bmax'] + 1):
			succes = Ed - c
			if succes < 0:
				break
			denomin_sum = denomin_sum + binom.pmf(succes, c, params['p3']) * lambda_**c * np.exp(-lambda_) / factorial(c)


		Distr_c_abd = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			succes = Ed - c
			if succes < 0:
				break
			Distr_c_abd[c] = binom.pmf(succes, c, params['p3']) * lambda_**c * np.exp(-lambda_) / factorial(c) 

		Distr_c_abd = Distr_c_abd / denomin_sum

		return Distr_c_abd, np.arange(Distr_c_abd.shape[0])

	if model == 1:
		
		a = Ea
		b = Eb
		
		# Precalculate vector of a - sums 
		A_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			A_sums[c] = binom.pmf(c, a, params['p1'])

		# Precalculate vector of b - sums 
		B_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		threshold = 5
		for c in range(0, params['amax'] + params['bmax'] + 1):
			if c > params['bmax']:
				break
			B_sums[c] = binom.pmf(c, b, params['p2'])
		
		# calculate distibution for c
		C_sums = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			sums = 0.0
			for cs in range(0, c+1):
				if A_sums[cs] == 0:
					continue
				sums = sums + A_sums[cs] * B_sums[c - cs]
			C_sums[c] = sums


		denomin_sum = 0.0
		for c in range(0, params['amax'] + params['bmax'] + 1):
			succes = Ed - c
			if succes < 0:
				break
			denomin_sum = denomin_sum + binom.pmf(succes, c, params['p3']) * C_sums[c]

		Distr_c_abd = np.zeros(params['amax'] + params['bmax'] + 1)
		for c in range(0, params['amax'] + params['bmax'] + 1):
			succes = Ed - c
			if succes < 0:
				break
			Distr_c_abd[c] = binom.pmf(succes, c, params['p3']) * C_sums[c] 

		Distr_c_abd = Distr_c_abd / denomin_sum

		return Distr_c_abd, np.arange(Distr_c_abd.shape[0])















