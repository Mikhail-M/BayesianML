from br_kovalenko import *

# Homework parametrs
params = {'amin':75, 'amax':90, 'bmin':500, 'bmax':600, 'p1':0.1, 'p2':0.01, 'p3':0.3}
# Ea = 83
# Eb = 550
# Ed = 18


# Calculate moments for given distribution
tic()
Distribution, support = pc_abd(params = params, model = 2)
toc()


E = np.dot(support, Distribution)
print "Expectation = " + str(E)


VAR = np.dot((support - E)**2, Distribution)
print "Variance = " + str(VAR) 


print 'Sum over support = ' + str(np.sum(Distribution))


# Plot distribution 
plt.xlabel('support')
plt.ylabel('P')
plt.title('Distribution')

plt.grid(True)
plt.plot(support, Distribution)
plt.show()




