import numpy as np
import matplotlib.pyplot as plt

# Say we have dy/dx = f(x)


# Say f is the equation for a bell curve
def fn(x):
    return (1/np.sqrt(np.pi*2))*np.exp(-x**2/2)


# plot our function
time = np.linspace(-5, 5, 1000)
data = fn(time)
plt.figure()
plt.plot(time, data)


# Integrate our function with euler's method
time_step = time[1] - time[0]
result = np.zeros(len(time))
result[0] = 0
for i in range(len(time) - 1):
    result[i+1] = result[i] + fn(time[i])*time_step


# plot the result
plt.figure()
plt.plot(time, result)
plt.show()