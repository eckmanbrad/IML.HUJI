from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

# ****** The following functions drop incoherent samples for learning ******


def __process_dates(data):
    years, months, days = [], [], []
    indices_to_drop = []
    for i, date in enumerate(data["date"]):
        if re.match(r"201\d[0-1]\d[0-3]\dT000000", str(date)):
            year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])
            years.append(year), months.append(month), days.append(day)
        else:
            indices_to_drop.append(i)

    data.drop(indices_to_drop, inplace=True, axis=0)
    data['year'] = years
    data['month'] = months
    data['day'] = days
    data.drop("date", inplace=True, axis=1)


def __drop_non_positive_samples_at(columns, data):
    for column in columns:
        data.drop(data[data[column] <= 0].index, inplace=True)


def __drop_non_negative_samples_at(columns, data):
    for column in columns:
        data.drop(data[data[column] < 0].index, inplace=True)


def __check_ranges_at(columns, ranges, data):
    for column, r in zip(columns, ranges):
        start, end = r
        data.drop(data[(data[column] > end) | (data[column] < start)].index, inplace=True)


def __process_year_built(data):
    indices_to_drop = []
    for i, year in enumerate(data["yr_built"]):
        if re.match(r"(19|20)\d{2}", str(int(year))) is None:
            indices_to_drop.append(i)
    data.drop(indices_to_drop, inplace=True, axis=0)


def __process_year_renovated(data):
    indices_to_drop = []
    for i, year in enumerate(data["yr_built"]):
        if year == 0: continue
        if re.match(r"(19|20)\d{2}", str(int(year))) is None:
            indices_to_drop.append(i)
    data.drop(indices_to_drop, inplace=True, axis=0)


def __process_zipcodes(data):
    indices_to_drop = []
    for i, zipcode in enumerate(data["zipcode"]):
        if zipcode == 0: continue
        if re.match(r"98\d{3}", str(int(zipcode))) is None:
            indices_to_drop.append(i)
    data.drop(indices_to_drop, inplace=True, axis=0)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    data = full_data[["date",
                      "price",
                      "bedrooms",
                      "bathrooms",
                      "sqft_living",
                      "sqft_lot",
                      "floors",
                      "waterfront",
                      "view",
                      "condition",
                      "grade",
                      "sqft_above",
                      "sqft_basement",
                      "yr_built",
                      "yr_renovated",
                      "zipcode",
                      "sqft_living15",
                      "sqft_lot15"]]

    # Clean
    __process_dates(data)
    __drop_non_positive_samples_at(["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "sqft_living15", "sqft_lot15"], data)
    __drop_non_negative_samples_at(["sqft_above", "sqft_basement"], data)
    __check_ranges_at(["waterfront", "view", "condition", "grade"], [(0,1), (0,4), (1,5), (1,13)], data)
    __process_year_built(data)
    __process_year_renovated(data)
    __process_zipcodes(data)
    # Quantify locations
    data = pd.get_dummies(data, columns=["zipcode"])
    # Scale data
    # data = pd.DataFrame(MinMaxScaler().fit_transform(data), index=data.index, columns=data.columns)
    # Clean labels
    prices = data[["price"]]
    # Drop labels from data
    data.drop("price", inplace=True, axis=1)
    return data, prices["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        pearson_cor = (np.cov(X[feature], y) / (np.std(X[feature])*np.std(y)))[0][1]
        fig = go.Figure()
        fig.add_scatter(x=X[feature], y=y, mode="markers")
        fig.update_layout(title=f"Price as a function of {feature} (correlation coefficient = {round(pearson_cor, 3)})")
        fig.write_image(f"{output_path}/{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, prices, r"C:\Users\baruc\OneDrive\Documents\HebrewU\Introduction to Machine Learning - 67577\ex2\correlation_plots")
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, prices)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # Train and predict
    estimator = LinearRegression()
    mean_losses, std_losses = [], []
    for p in range(10, 101):
        percentage = p/100
        losses = []
        for i in range(10):
            samples_train_X, samples_train_y, samples_test_X, samples_test_y = split_train_test(train_X, train_y, percentage)
            estimator.fit(np.array(samples_train_X), np.array(samples_train_y))
            losses.append(estimator.loss(np.array(test_X), np.array(test_y)))
        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))
    # Plot
    mean_losses, std_losses = np.array(mean_losses), np.array(std_losses)
    x = list(range(10, 101))
    plot1 = go.Scatter(x=x, y=mean_losses, mode="markers+lines",
                       name="Mean Loss", line=dict(dash="dash"), marker=dict(color="green", opacity=.7))
    plot2 = go.Scatter(x=x, y=mean_losses - 2 * std_losses, fill=None, mode="lines",
                       line=dict(color="lightgrey"), showlegend=False)
    plot3 = go.Scatter(x=x, y=mean_losses + 2 * std_losses, fill='tonexty', mode="lines",
                       line=dict(color="lightgrey"), showlegend=False)
    fig = go.Figure()
    fig.add_trace(plot1)
    fig.add_trace(plot2)
    fig.add_trace(plot3)
    fig.update_layout(title_text="Mean loss as a function of the percentage of samples trained on",
                      xaxis_title="Percentage of samples", yaxis_title="Mean Loss")
    fig.show()
