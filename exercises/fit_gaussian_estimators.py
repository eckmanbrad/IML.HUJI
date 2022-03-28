from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    estimator = UnivariateGaussian()
    mu, var, sample_size = 10, 1, 1000
    samples = np.random.normal(mu, var, sample_size)
    estimator.fit(samples)
    print("({},{})".format(estimator.mu_, estimator.var_))

    # Question 2 - Empirically showing sample mean is consistent
    x = [i for i in range(10, 1001, 10)]
    y = [0.0]*100
    for i in range(100):
        estimator.fit(samples[:(i+1)*10])
        y[i] = abs(estimator.mu_ - mu)
    px.line(x=x, y=y, labels={"x": "Sample size", "y": "Estimator distance from expectation"},
            title="Deviance from actual expectation as a function of sample size (samples taken from a univariate "
                 "Gaussian X~N(10,1))").show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = estimator.pdf(samples)
    df2 = pd.DataFrame({"Observed samples": samples, "PDF": pdf})
    px.scatter(df2, x="Observed samples", y="PDF", title="PDF as a function "
                "of observed samples from a univariate Gaussian X~N(10,1)").show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    estimator = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([1, 0.2, 0, 0.5,
                    0.2, 2, 0, 0,
                    0, 0, 1, 0,
                    0.5, 0, 0, 1]).reshape(4, 4)
    sample_size = 1000
    samples = np.random.multivariate_normal(mu, cov, sample_size)
    estimator.fit(samples)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    numbers = np.linspace(-10, 10, 200)
    data = np.array([[0.0]*200 for i in range(200)])
    for i, f1 in enumerate(numbers):
        for j, f3 in enumerate(numbers):
            mu = np.array([f1, 0, f3, 0])
            data[i][j] = MultivariateGaussian.log_likelihood(mu, cov, samples)

    fig = px.imshow(data, x=numbers, y=numbers)
    fig.layout.update(title="Log-likelihood for mu=(f1, 0, f3, 0) and fixed covariance")
    fig.layout["xaxis1"].update(title="f3")
    fig.layout["yaxis1"].update(title="f1")
    fig.show()

    # Question 6 - Maximum likelihood
    argmax = data.argmax()
    print("f1: {}, f3: {}".format(numbers[argmax // 200], numbers[argmax % 200]))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


