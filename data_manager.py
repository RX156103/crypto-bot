import pandas as pd
from datetime import datetime, date, timedelta
import h5py as h5
import numpy as np
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set()

# data parser function shortcut
dp = date.fromtimestamp
dtp = datetime.fromtimestamp

use = 's14_20180810_20190107'


def prepare_data(file, prune=False):
    if prune:
        # initialize the data
        data = pd.read_csv("data/{}.csv".format(file), names=['time', 'price', 'delta'])
        # remove duplicates and refresh the data
        data.drop_duplicates('time', keep=False, inplace=True)
        data = data[data['time'] > datetime(2015, 1, 1).timestamp()]
        data.to_csv("data/{}.csv".format(file), index=False)
        # free data from memory
        del data

    print('done with initializiation of data set')

    # create hdf5 file
    output_file = h5.File('data/{}_data.hdf5'.format(file), 'w')

    all_data_cols = output_file.create_group('data')
    csv_file = open('data/{}.csv'.format(file))

    # count number of entries; exclude header row
    line_count = -1
    for _ in csv_file:
        line_count += 1

    csv_file.seek(0)  # start iterator at the beginng of the file
    lines = iter(csv_file)

    # grab the header column data
    header = next(lines)
    cols = header.strip().split(',')

    # initialize the all data group
    Nbatch = 10 ** 4
    all_datasets, numpy_arrs = [], []
    for col in cols:
        dataset = all_data_cols.create_dataset(col, (line_count,), dtype='f8')
        all_datasets.append(dataset)
        numpy_arrs.append(np.zeros((Nbatch,), dtype='f8'))

    # read in Nbatch lines at a time and write them out
    row = 0

    # batch variables
    tmp_arr, ct = [], 0
    current_date, previous_timestamp = None, None
    create_new = True
    threshold = 1000

    # iterate through all lines
    for line in lines:
        # convert line to a series of float values
        values = list(map(float, line.split(',')))

        if row % Nbatch == 0 and row > 0:
            # print out data
            for i in range(len(cols)):
                start = (row // Nbatch - 1) * Nbatch
                end = (row // Nbatch) * Nbatch
                all_datasets[i][start:end] = numpy_arrs[i][:]
        # load in all regular data
        for i in range(len(cols)):
            index = row % Nbatch
            numpy_arrs[i][index] = values[i]

        # handle subsets

        # initialize variables
        if create_new:
            current_date, previous_timestamp, tmp_arr = dp(values[0]), values[0], []
            tmp_arr.append(values)
            create_new = False
        else:
            # check if the difference is too large
            diff_ts = values[0] - previous_timestamp

            # difference is greater than threshold, push data into data stream
            if (diff_ts >= threshold and dp(values[0]) != current_date) or row == line_count - 1:
                if len(tmp_arr) > 1000:
                    key = 's{}_{}_{}'.format(ct, dp(tmp_arr[0][0]).strftime('%Y%m%d'),
                                             dp(values[0]).strftime('%Y%m%d'))
                    h5columns = output_file.create_group('subsets/{}'.format(key))
                    for i in range(len(cols)):
                        ds = (h5columns.create_dataset(cols[i], (len(tmp_arr),), dtype='f8'))
                        ds[:] = [item[i] for item in tmp_arr]
                    ct += 1
                    print("name: {} ({}) size: {} left: {}".format(key, ct, len(tmp_arr), line_count - (row + 1)))
                create_new = True
            else:
                tmp_arr.append(values)
                previous_timestamp = values[0]
                current_date = dp(values[0])

        row += 1

    # fill left overs
    if (row % Nbatch) > 0:
        for i in range(len(cols)):
            start = (int(row / Nbatch)) * Nbatch
            end = line_count
            all_datasets[i][start:end] = numpy_arrs[i][:end - start]


Interval = Enum('Interval', 'minute hour day')


def generate_intervals(file, subset=None, interval=Interval.minute):
    comp = lambda a, b: a.minute != b.minute
    if interval == Interval.hour:
        comp = lambda a, b: a.hour != b.hour
    elif interval == Interval.day:
        comp = lambda a, b: a.day != b.day

    df = pd.HDFStore('data/{}_data.hdf5'.format(file))
    out = h5.File('data/{}_{}.hdf5'.format(file, interval.name), 'w')

    # pick the subset
    subsets = df.root['subsets']
    if subset is not None:
        subsets = [df.root['subsets/{}'.format(subset)]]

    # iterate through each subset
    for subset in subsets:
        data_cols = out.create_group('data/' + subset._v_name)
        arr, lt = [], dtp(subset['time'][0])
        o, c = subset['price'][0], subset['price'][0]
        # loop through all entries in subset
        for i in range(1, len(subset['time'])):
            ct = dtp(subset['time'][i])
            # if everything is the same but the minute, push data in and switch to next minute
            if comp(ct, lt) or i == len(subset['time']) - 1:
                c = subset['price'][i - 1]
                # strip seconds from datetime and create timestamp
                time = datetime(year=lt.year, month=lt.month, day=lt.day, hour=lt.hour, minute=lt.minute)
                arr.append([time.timestamp(), o, c])

                # set up next items
                lt = ct
                o = subset['price'][i]
        columns = ['time', 'open', 'close']
        for i in range(len(columns)):
            x = data_cols.create_dataset(columns[i], shape=(len(arr),), dtype='f8')
            x[:] = [tmp[i] for tmp in arr]
        print('done with set: {}'.format(subset._v_name))
    out.close()


def load_interval(file, interval=Interval.minute):
    df = pd.HDFStore('data/{}_{}.hdf5'.format(file, interval.name)).root['data']
    for item in df:
        dates = [dtp(t) for t in item['time']]
        t = pd.DataFrame(dates)
        ct = count_out_of_seq(dates, interval)
        print(ct)


# count the number of times the list was out of sequence
def count_out_of_seq(items, interval=Interval.minute):
    ct = 0
    d = timedelta(minutes=1)
    if interval == Interval.hour:
        d = timedelta(hours=1)
    elif interval == Interval.day:
        d = timedelta(days=1)

    for i in range(1, len(items)):
        ct += 1 if items[i - 1] + d != items[i] else 0
    return ct


# compute rsi and store
def compute_rsi(n=40):
    df: pd.DataFrame = pd.read_feather('data/training_data.feather')
    prices = df['price']
    deltas = np.diff(prices)
    seed = deltas[:n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100. / (1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    df['rsi'] = rsi
    df.to_feather('data/training_data.feather')
    return rsi


# compute macd and store
def compute_macd(fast=12, slow=26, smooth=9):
    df = pd.read_feather('data/training_data.feather')
    prices = df['price']

    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()

    macd = exp1 - exp2
    exp3 = macd.ewm(span=smooth, adjust=False).mean()

    df['macd'] = macd
    df['macd_signal'] = exp3
    df.to_feather('data/training_data.feather')


# plot relevant data for time period
def plot_data(start=0, end=None):
    df = pd.read_feather('data/training_data.feather')
    if end is None:
        end = len(df)
    df = df[start:end]
    times = df['time'].apply(lambda t: dtp(t))

    fig = plt.figure(figsize=(8, 13))
    title = fig.suptitle("Bitcoin Data ({} - {})".format(times.iloc[0], times.iloc[-1]))

    price_ax: plt.Axes = fig.add_subplot(3, 1, 1)
    price_ax.set_xlabel('Time')
    price_ax.set_ylabel('Dollar Value')
    price_plt = price_ax.plot(times, df['price'], color='#5DADE2', label='Price')
    price_ax.legend(loc='upper left')

    macd_ax: plt.Axes = fig.add_subplot(3, 1, 2)
    macd_ax.set_xlabel('Time')
    macd_ax.set_ylabel('MACD Value')
    macd_plt = macd_ax.plot(times, df['macd'], label='MACD', color='#3498DB')
    signal_plt = macd_ax.plot(times, df['macd_signal'], label='Signal Line', color='#E5A4CB')
    macd_ax.legend(loc='upper left')

    rsi_ax: plt.Axes = fig.add_subplot(3, 1, 3)
    rsi_ax.set_xlabel('Time')
    rsi_ax.set_ylabel('RSI Value')
    rsi_plt = rsi_ax.plot(times, df['rsi'], color='#DB344B', label='RSI')
    rsi_ax.legend(loc='upper left')
    plt.savefig('images/bitcoin_data.png')

    plt.show()


plot_data(0, 1000)
print('done')
