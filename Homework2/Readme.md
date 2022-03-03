26 Feb 2022

**Wtech AI Course - Homework - 2**

**Deadline: 3rd March 2022 23:59![](Aspose.Words.471429df-f5ad-4814-b97c-55cb8274d284.001.png)**

**Submission:** Submit your code/ jupyter notebook to <mursel.tasgin@gmail.com>

**Important Note:** If you have not opened a github account on github.com yet, please open it and create a repository. Put all of your homeworks also on your github repo, add me as a contributor to your repo (my github profile is murselTasginBoun)

**Homework Details:![](Aspose.Words.471429df-f5ad-4814-b97c-55cb8274d284.002.png)**

Implement linear regression from scratch without using a library *(do not use scikit-learn’s regression function or any other library’s regression function).* You can use numpy, pandas, matplotlib or other helper libraries.

Details:

- **First generate some data points as your input data for regression. You can use the following function or develop your own function for data generation** def generate\_data(n, beta\_0, beta\_1):

x = np.arange(n)

e = np.random.uniform(-10,10, size=(n,))

y = beta\_0 + beta\_1\* x + e

return x,y

i.e., you can call this function to create 100 data points (x,y) x, y = generate\_data(100, 2, .4)

- **Find the line that fits best to the generated data. Use MSE (mean root square error) as your cost function.**
- **Plot the line you find using your regression function. You can use matplotlib or any python plotting library.**
- **Plot the MSE or RMSE of your regression function during iterations.**
