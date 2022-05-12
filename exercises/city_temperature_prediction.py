import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    # data = full_data[["Country", "City", "Date", "Year", "Month", "Day", "Temp"]]
    data.drop(data[data["Temp"] < -5].index, inplace=True, axis=0)
    # data.drop("Temp", inplace=True, axis=1)
    # temps = data[["Temp"]]
    data["DayOfYear"] = [pd.Period(date, freq='H').day_of_year for date in data["Date"]]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data.loc[data["Country"] == "Israel"]
    israel_data["Year"] = israel_data["Year"].astype(str)  # convert to string
    fig1 = px.scatter(israel_data, x="DayOfYear", y="Temp", color="Year",
                     title="Temperature in Israel as a function of the DayOfYear")
    fig1.show()
    israel_data["Year"] = israel_data["Year"].astype(int)  # convert to int

    agg_israel_temps_std = israel_data.groupby("Month").Temp.agg('std')
    grouped_israel_data = pd.DataFrame({"Month": list(range(1, 13)),
                                        "Temp std": list(agg_israel_temps_std)})
    fig2 = px.bar(grouped_israel_data, x="Month", y="Temp std", title="Standard deviation of temperatures in Israel")
    fig2.show()

    # Question 3 - Exploring differences between countries
    df = pd.DataFrame({"Country": [], "Month": [], "Temp mean": [], "Temp std": []})
    for country in data["Country"].unique():
        country_data = data.loc[data["Country"] == country]
        temp_means = country_data.groupby("Month").Temp.agg('mean')
        temp_stds = country_data.groupby("Month").Temp.agg('std')
        country_df = pd.DataFrame({"Country": [country]*12,
                                   "Month": list(range(1, 13)),
                                   "Temp mean": temp_means,
                                   "Temp std": temp_stds})
        df = df.append(country_df, ignore_index=True)

    fig3 = px.line(df, x="Month", y="Temp mean", color="Country", error_y="Temp std", title="Average temperature by month")
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    losses = []
    train_X, train_y, test_X, test_y = split_train_test(israel_data["DayOfYear"], israel_data["Temp"])
    for k in range(1, 11):
        estimator = PolynomialFitting(k).fit(np.array(train_X), np.array(train_y))
        loss = round(estimator.loss(np.array(test_X), np.array(test_y)), 2)
        print(f"Loss for polynomial fitting of degree {k}: {loss}")
        losses.append(loss)

    df = pd.DataFrame({"Degree of polynomial fitted": list(range(1, 11)),
                      "Loss": losses})
    fig4 = px.bar(df, x="Degree of polynomial fitted", y="Loss", title="Loss as a function of the degree of the fitted polynomial")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    losses = []
    israel_estimator = PolynomialFitting(5).fit(np.array(israel_data["DayOfYear"]), np.array(israel_data["Temp"]))
    for country in data["Country"].unique():
        country_data = data.loc[data["Country"] == country]
        loss = round(israel_estimator.loss(np.array(country_data["DayOfYear"]), np.array(country_data["Temp"])), 2)
        losses.append(loss)
    df = pd.DataFrame({"Country": [country for country in data["Country"].unique()],
                       "Loss": losses})
    fig5 = px.bar(df, x="Country", y="Loss",
                  title="Prediction error of model trained on Israel data")
    fig5.show()
