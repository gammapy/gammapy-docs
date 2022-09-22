
# coding: utf-8

# # Rapid introduction on using numpy, scipy, matplotlib
# 
# This is meant to be a very brief reminder. It is strongly suggested to refer to more detailed
# introductions and tutorials see for instance:
# - [A Whirlwind tour of Python](http://nbviewer.jupyter.org/github/jakevdp/WhirlwindTourOfPython/blob/master/Index.ipynb)
# - [Python data science handbook](http://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb)
# - [Scipy lectures](http://www.scipy-lectures.org/)
# 
# ## Introduction
# 
# Here we will look at :
# - basic features regarding array manipulation and indexing
# - do a bit of plotting with matplotlib
# - use a number of useful scipy features
# - see an example of vectorization with a simple Monte Carlo problem

# ## numpy: arrays, indexing etc
# 
# 

# In[1]:


import numpy as np


# In[2]:


np.array([3,4,5])


# In[3]:


np.array([[1, 2],[3,4]])


# In[4]:


### linearly spaced 1D array
np.linspace(1.,10.,10)


# In[5]:


### log spaced 1D array
np.logspace(0.,1.,10)


# In[6]:


### 1D array of zeros
np.zeros(5)


# In[7]:


### 2D array of zeros
np.zeros((3,3))


# ### Types and casts
# 
# See numpy [dtypes](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)

# In[8]:


x_int = np.logspace(0.,1.,10).astype('int')   # cast array as int
print(x_int)


# In[9]:


x_int[1] = 2.34   # 2.34 is cast as int
print(x_int[1])


# In[10]:


array_string = np.array(['a','b','c','d'])
array_string.dtype    # 1 character string


# In[11]:


array_string[1]='bbbb'   # 'bbbb' is cast on 1 character string
array_string[1]


# In[12]:


array_string = np.array(['a','b','c','d'],dtype=np.dtype('S10'))
array_string[1] = 'bbbb'   # 'bbbb' is cast on 10 character string
array_string[1]


# ### array indexing & slicing

# In[13]:


x = np.arange(10)


# In[14]:


x[-1]   # last element


# In[15]:


x[3:6]  # subarray


# In[16]:


x[1::2] # stride


# In[17]:


x[::-1] # stride


# In[18]:


x = np.array([np.arange(10*i,10*i+5) for i in range(5)])
x


# In[19]:


print("first column : ", x[:,0])
print("last row     : ", x[-1,:])


# In[20]:


b=x[-1,:]   # This is a view not a copy!
b[:] += 1

print(x) # the initial matrix is changed!


# In[21]:


# Fancy indexing 
print(x % 2 == 1)


# In[22]:


x[x % 2 == 1] = 0
print(x)


# ### Broadcasting

# In[23]:


x = np.linspace(1, 5, 5) + 4   # 4 is broadcast to 5 element array
x


# In[24]:


y = np.zeros((3, 5)) + x   # x is broadcast to (3,5) array
y


# ## Plotting with matplotlib
# 
# We will see some plotting:
# - Simple plots
# - Histograms with matplotlib

# In[25]:


# This is for embedding figures in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')         # Fancy style


# ## Vectorization or loops: A very simple MC
# 
# We want to solve a simple statistical question. Assume a Poisson random process of mean mu. What is the density probability function pdf(n_val) of having at least one realization of the Poisson process out of N larger than n_val? 
# 
# See for instance [this paper](https://arxiv.org/pdf/0903.4373.pdf)
# 
# While this problem has an analytical solution we would like to test it with a simple MC. 
# 
# We will first do it as one would do it with a C code and we will progressively vectorize the problem. We will use a timer to compare performance.
# 
# 

# In[26]:


### Define the function
def poisson_sample_maximum(mu, N, Ntrials):
    """
    Generate a set of Ntrials random variables defined as the maximum of N 
    random Poisson R.V. of mean mu
    """
    res = np.zeros(Ntrials)
    ### Do a loop
    for i in range(Ntrials):
        ### Generate N random varslues 
        Y = np.random.poisson(mu, size=(N))
        ### Take the maximum 
        res[i] = np.max(Y)

    return res 
   
mu = 5
N = 10
Ntrials = 1000000
    
get_ipython().run_line_magic('timeit', 'values = poisson_sample_maximum(mu, N, Ntrials)')


# It does work, but no so fast...
# 
# To do it in a efficient and pythonic way we have to avoid loops as much as possible.
# 
# The idea here will then be to do all trials at once requiring random.poisson to produce a 2D matrix of size Nxtrials

# In[27]:


### Define a better function
def poisson_sample_maximum_better(mu, N, Ntrials):
    """
    Generate a set of Ntrials random variables defined as the maximum of N 
    random Poisson R.V. of mean mu
    """
    ### Generate N*Ntrials random values in N x Ntrials matrix
    Y = np.random.poisson(mu,size=(N,Ntrials))
    ### Return the maximum in each row
    return np.max(Y,0)
   
mu = 5
N = 10
Ntrials = 1000000
    
get_ipython().run_line_magic('timeit', 'values = poisson_sample_maximum_better(mu, N, Ntrials)')


# We can now compare the distribution of MC simulated values to the actual analytical function.
# 

# In[28]:


values = poisson_sample_maximum_better(mu,N,Ntrials)

### Make and plot the normalized histogram
### We define the binning ouselves to have bins for each integer
bins = np.arange(0, 10 * mu)
histo = plt.hist(values, bins=bins, normed=True, log=True)

### Now compare to the analytical solution
from scipy.special import gammaincc

### Define a lambda function to compute analytical solution
proba = lambda nv, Nr, mu_p : gammaincc(nv + 1, mu_p) ** Nr - gammaincc(nv, mu_p) ** Nr

x = 0.5 * (bins[:-1] + bins[1:])
y = proba(bins[:-1], N, mu)
plt.plot(x, y)
plt.ylim(1e-6,1)   # restrict y range


# ## Exercices
# 
# - write a vectorized function that takes an array of int and returns an array where square integers are replaced by their square root and the others are left unchanged

# In[29]:


### A solution
def replace_square(n):
    sqrt_n = np.sqrt(n)
    return n + (sqrt_n == sqrt_n.astype(int))*(-n + sqrt_n)

print(replace_square(7.0))
print(replace_square(np.arange(26)))


# In[30]:


### or using where
def replace_square2(n):
    sqrt_n = np.sqrt(n)
    return np.where(sqrt_n == sqrt_n.astype(int), 
                    sqrt_n, n)
        
print(replace_square2(7.0))       
print(replace_square2(np.arange(26)))

