=========
sagemaker
=========


Add a short description here!
Good.

>How to run program

```$ conda create -n boston_housing python=3.6 jupyterlab pandas scikit-learn seaborn```


Description
===========

A longer description of your project goes here...


Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.

"""Data Science Challenge for C1"""

# import requests, zipfile, io, json
# import pandas as pd
# from pandas import DataFrame, read_csv
import numpy as np
# import os
# from functools import reduce
# import datetime as dt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import metrics
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
# import matplotlib.pyplot as plt
# from matplotlib.pylab import rcParams
# import seaborn as sns
#
# rcParams['figure.figsize'] = 6, 8
# plt.style.use('ggplot')
# sns.set_style("whitegrid")


def color_unique_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if val.isinstance(int):
        color = 'red'
    else:
        color = 'black'
    return 'color: %s' % color


def convert_trans_datetime(data):
    """Missing function docstring TODO"""
    data['transactionDateTime'] = pd.to_datetime(
        data['transactionDateTime'], format="%Y-%m-%dT%H:%M:%S")
    data['accountOpenDate'] = pd.to_datetime(data['accountOpenDate'],
                                             format="%Y-%m-%d")
    data['dateOfLastAddressChange'] = pd.to_datetime(
        data['dateOfLastAddressChange'], format="%Y-%m-%d")
    return data


# Get column list by dropping some columns
def get_col_list(df, drop_list):
    """TODO"""
    feature_list = list(df.columns)
    remain_list = [item for item in feature_list if item not in drop_list]
    return remain_list


# Get dataframe by dropping some columns
def drop_col(df, drop_list):
    """TODO"""
    remain_list = get_col_list(df, drop_list)
    return df[remain_list]


# Filter trans by two trans time diff
def filter_time_diff(data, low, high):
    """TODO"""
    data = data[(data.time_diff > low) & (data.time_diff < high)]
    return data


# Filter trans by two trans relationship
# The former trans' availableMoneye - transactionAmount= the
# latter one's availableMoney,
# since charged again.
def filter_trans_du(data):
    """TODO"""
    data = data[data.availableMoney_x -
                data.transactionAmount == data.availableMoney_y]
    return data


# Get the duplicate transactions
# df: input data frame
# not_iden_feat: not identical features for two trans
# low, high: range of time diff
# style: 'F' is full dataframe; while 'S' is simplified dataframe
def get_duplicate(df, not_iden_feat, low, high, style='F'):
    """TODO"""
    # Get dataframe whose transactionType is not REVERSAL
    df_DUP = df[df['transactionType'] != 'REVERSAL']
    # Print transactionDateTime stat
    print('transactionDateTime STAT: \n {} \n'.format(
        df_DUP['transactionDateTime'].describe()))

    # Sort dataframe by accountNumber and transactionDateTime
    df_sort = df_DUP.sort_values(by=['accountNumber', 'transactionDateTime'])
    # Group sorted datafram by features in du_list
    du_list = ['accountNumber', 'transactionAmount',
               'acqCountry', 'accountOpenDate',
               'cardCVV', 'cardLast4Digits',
               'cardPresent', 'creditLimit', 'currentExpDate',
               'dateOfLastAddressChange',
               'expirationDateKeyInMatch', 'isFraud',
               'merchantCategoryCode', 'merchantCountryCode',
               'merchantName', 'posConditionCode', 'posEntryMode',
               'enteredCVV']
    # Group by du_list since the features of two trans
    # in du_list should be same.
    df_grouped = pd.concat(g for _, g in df_sort.groupby(du_list)
                           if len(g) > 1)
    # Print grouped dataframe shape
    print('grouped dataframe shape: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_grouped.shape[0],
                                           df_grouped.shape[1]))

    # Join two identical grouped dataframe
    iden_feat_list = get_col_list(df_grouped, not_iden_feat)
    df_merged = pd.merge(df_grouped, df_grouped, on=iden_feat_list,
                         how='inner')
    # Print merged dataframe shape
    print('merged dataframe shape: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_merged.shape[0],
                                           df_merged.shape[1]))

    # Add time diff between two trans in dataframe
    df_merged['time_diff'] = (df_merged['transactionDateTime_y'] -
                              df_merged['transactionDateTime_x']
                              ) / np.timedelta64(1, 's')
    # Print time diff stat
    print('time diff between two trans in dataframe STAT: \n',
          df_merged['time_diff'].describe())

    # Filter rows by time diff
    df_du = filter_time_diff(df_merged, low, high)
    # Print dataframe shape after filtering by time diff
    print('dataframe shape after filtering by time diff: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_du.shape[0], df_du.shape[1]))

    # Filter rows by two transactions relationship
    df_du = filter_trans_du(df_du)
    # Print dataframe shape after filtering by trans relationshp
    print('dataframe shape after filtering by trans relationshp: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_du.shape[0], df_du.shape[1]))

    if style == 'S':
        # Output simplified final dataframe
        # v_list is the 'not same features' list in final df
        v_list = get_col_list(df_du, iden_feat_list)
        l_to_add = ['transactionAmount', 'transactionType']
        v_list.extend(l_to_add)
        df_du_f = df_du[v_list]
    elif style == 'F':
        df_du_f = df_du
    return df_du_f


def get_report_du(df, not_iden_feat):
    """TODO"""
    no_of_trans = df.shape[0]
    amount = df['transactionAmount'].sum()
    dist = df['transactionType'].value_counts()
    print('The time different between transactions datatime: '
          '{} to {} Seconds.'.format(low, high))
    print('Not identical features in two transactions: ', not_iden_feat)
    print('Consider the first transaction to be "normal" and \
    exclude it from the number of transaction and dollar amount \
    counts. Number of transactions is {} and the dollar amount is {}. \
    The transactionType Distribution is: \n {}'.
          format(no_of_trans, amount, dist))
    return


def filter_time_diff(data, low, high):
    """TODO"""
    data = data[(data.time_diff > low) & (data.time_diff < high)]
    return data


def filter_trans_rev(data):
    """TODO"""
    #     data = data[data.currentBalance_x + data.transactionAmount
    # == data.currentBalance_y]
    data = data[data.availableMoney_x + data.transactionAmount
                == data.availableMoney_y]
    return data


def get_rev(df, not_iden_feat, low, high, style='F'):
    """TODO"""
    # Create dataframe whose transactionType is REVERSAL
    df_REVERSAL = df[df['transactionType'] == 'REVERSAL']
    # Print shape of dataframe whose transactionType is REVERSAL
    print('shape of dataframe whose transactionType is REVERSAL: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_REVERSAL.shape[0],
                                           df_REVERSAL.shape[1]))
    # Print distribution
    print('Distribution by transactionType: \n',
          df_REVERSAL['transactionType'].value_counts())

    # Create dataframe whose transactionType is not REVERSAL
    df_not_REVERSAL = df[df['transactionType'] != 'REVERSAL']
    # Print shape of dataframe whose transactionType is not REVERSAL
    print('shape of dataframe whose transactionType is not REVERSAL: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_not_REVERSAL.shape[0],
                                           df_not_REVERSAL.shape[1]))
    # Print distribution
    print('Distribution by transactionType: \n',
          df_not_REVERSAL['transactionType'].value_counts())

    # Select features used to identify similar transactions
    iden_feat_list = ['accountNumber', 'transactionAmount', 'acqCountry',
                      'accountOpenDate', 'cardCVV', 'cardLast4Digits',
                      'cardPresent', 'creditLimit', 'currentExpDate',
                      'dateOfLastAddressChange', 'expirationDateKeyInMatch',
                      'isFraud', 'merchantCategoryCode', 'merchantCountryCode',
                      'merchantName', 'posConditionCode', 'posEntryMode',
                      'enteredCVV']

    # Merge two dataframes by same features
    df_merged = pd.merge(df_not_REVERSAL, df_REVERSAL,
                         on=iden_feat_list, how='inner')
    # Print merged dataframe shape
    print('merged dataframe shape: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_merged.shape[0],
                                           df_merged.shape[1]))

    # Print distribution not REVERSAL
    print('Distribution of (transactionType=not_REVERSAL): \n',
          df_merged['transactionType_x'].value_counts())
    # Print distribution REVERSAL
    print('Distribution of (transactionType=REVERSAL): \n',
          df_merged['transactionType_y'].value_counts())

    # Add time diff between two trans in dataframe
    df_merged['time_diff'] = (df_merged['transactionDateTime_y'] -
                              df_merged['transactionDateTime_x']
                              ) / np.timedelta64(1, 'D')
    # Print time diff stat
    print('time diff between two trans in dataframe STAT: \n',
          df_merged['time_diff'].describe())

    # Filter rows by time diff
    df_rev = filter_time_diff(df_merged, low, high)
    # Print dataframe shape after filtering by time diff
    print('dataframe shape after filtering by time diff: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_rev.shape[0], df_rev.shape[1]))

    # Filter rows by two transactions relationship
    df_rev = filter_trans_rev(df_rev)
    # Print dataframe shape after filtering by trans relationshp
    print('dataframe shape after filtering by trans relationshp: \
          Number of rows: {} \
          Number of columns: {} \n'.format(df_rev.shape[0], df_rev.shape[1]))

    # Print distribution not REVERSAL
    print('Distribution of (transactionType=not_REVERSAL): \n',
          df_rev['transactionType_x'].value_counts())
    # Print distribution REVERSAL
    print('Distribution of (transactionType=REVERSAL): \n',
          df_rev['transactionType_y'].value_counts())

    if style == 'S':
        # Output simplified final dataframe
        filter_list = ['transactionAmount', 'index_x', 'availableMoney_x',
                       'currentBalance_x', 'transactionDateTime_x',
                       'transactionType_x',
                       'index_y', 'availableMoney_y', 'currentBalance_y',
                       'transactionDateTime_y', 'transactionType_y',
                       'time_diff']
        df_rev_f = df_rev[filter_list]
        df_rev_f = df_rev

    elif style == 'F':
        df_rev_f = df_rev
    return df_rev_f


def get_report_rev(df):
    """TODO"""
    no_of_trans = df.shape[0]
    amount = df['transactionAmount'].sum()
    dist = df['transactionAmount'].value_counts()
    print('The time different between transactions datatime: '
          '{} to {} Days.'.format(low, high))
    print('Number of transactions is {} and the dollar amount is '
          '{}'.format(no_of_trans, amount))
    print('transactionAmount Distribution is {}'.format(dist))
    return


def splitDatetime(data):
    """TODO"""
    datatime = pd.DatetimeIndex(data.transactionDateTime)
    data['year'] = datatime.year
    data['month'] = datatime.month
    data['day'] = datatime.day
    data['hour'] = datatime.hour
    data['minute'] = datatime.minute
    data['second'] = datatime.second
    data['weekday'] = datatime.weekday
    return data


def create_feature(data):
    """TODO"""
    # transaction amount/ credit limit
    data['Amount_limit'] = data['transactionAmount'] / data['creditLimit']
    # available money/ credit limit
    data['available_limit'] = data['availableMoney'] / data['creditLimit']
    # current balance/ credit limit
    data['current_limit'] = data['currentBalance'] / data['creditLimit']
    # transaction date - account open date
    data['trans_open'] = (data['transactionDateTime'] -
                          data['accountOpenDate']
                          ) / np.timedelta64(1, 'D')
    # transaction date - date of last address change
    data['trans_address'] = (data['transactionDateTime'] -
                             data['dateOfLastAddressChange']
                             ) / np.timedelta64(1, 'D')
    return data


def process_data(df):
    """TODO"""
    data = df.copy()
    # 1. Remove unuseful features (null, nan, duplicate)
    drop_list = ['echoBuffer', 'merchantCity',
                 'merchantState', 'merchantZip',
                 'posOnPremises', 'recurringAuthInd',
                 'customerId']
    remain_list = get_col_list(data, drop_list)
    #     data = data.drop(['customerId'], axis=1)
    data = data[remain_list]
    # 2. Splite datetime
    data = splitDatetime(data)
    # 3. Create new features
    data = create_feature(data)
    return data


def encode_cat(data):
    """TODO"""
    l_not_number = list(data.columns.where(data.dtypes != np.number))
    cat_list = [x for x in l_not_number if str(x) != 'nan']
    for col in cat_list:
        data[col] = le.fit_transform(data[col])
    return data


def train_test_data(df, random_state):
    """TODO"""
    features = [item for item in list(df.columns) if item not in ['isFraud']]
    return train_test_split(df[features],
                            df['isFraud'],
                            test_size=0.2,
                            stratify=df['isFraud'],
                            random_state=random_state)


def modelfit(alg, dtrain, dtest, predictors, printFeatureImportance=True):
    """TODO"""
    label = 'isFraud'
    # Fit the algorithm on the data
    clf = alg.fit(dtrain[predictors], dtrain[label])

    # Predict training set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Number of Training Data : %d" % len(dtrain))
    print("Number of Testing Data : %d" % len(dtest))
    print("Accuracy : %.4g" % metrics.accuracy_score(
        dtest[label].values, dtest_predictions))
    print("AUC Score (Test): %f" % metrics.roc_auc_score(
        dtest[label], dtest_predprob))
    print("Confusion Matrix: \n", metrics.confusion_matrix(
        dtest[label].values, dtest_predictions, labels=[0, 1]))
    C = metrics.confusion_matrix(dtest[label].values,
                                 dtest_predictions, labels=[0, 1])
    show_confusion_matrix(C, ['Valid', 'Fraud'])

    # Print Feature Importance:
    if printFeatureImportance:
        rcParams['figure.figsize'] = 10, 10
        feat_imp = pd.Series(alg.feature_importances_,
                             predictors).sort_values(ascending=True)
        feat_imp.plot(kind='barh', title='Feature Importance')
        plt.ylabel('Feature')
    #         plt.savefig('if.png')
    return clf


def show_confusion_matrix(C, class_labels=['0', '1']):
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """

    rcParams['figure.figsize'] = 6, 6

    assert C.shape == (2, 2), "Confusion matrix should be " \
                              "from binary classification only."

    # true negative, false positive, etc...
    tn = C[0, 0]
    fp = C[0, 1]
    fn = C[1, 0]
    tp = C[1, 1]

    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.gray)

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f' % (fp / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Pre Val: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 2,
            'Pos Pred Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    plt.tight_layout()
    #     plt.savefig('cm.png')
    plt.show()
