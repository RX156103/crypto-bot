from collections import namedtuple
from enum import Enum
from abc import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

sns.set()

action = Enum('Action', 'buy sell hold')
transaction = namedtuple('Transaction', 'action time')

"""
Data to incorporate
* price derivative
* price second derivative
* rsi 
* rsi derivative
* macd
* macd derivative
"""


class TestEnv(ABC):
    _TRANSACTION_FEE = 0.000001

    def __init__(self, df, cash):
        if df is None:
            raise Exception("BRUH GIMME SOME REAL DATA")
        self._df = df
        self._cash, self._btc = cash, 0
        self._trades = []
        self.index = -1
        self.cash_history, self.transactions = [], []

    @property
    def btc_price(self):
        if self._df is not None:
            return self._df['price'][self.index]
        else:
            return -1

    @property
    def can_buy(self):
        return self._btc == 0 and self._cash > 0

    @property
    def can_sell(self):
        return self._btc != 0

    @property
    def rsi(self):
        return self._df['rsi'][self.index]

    @property
    def macd(self):
        return self._df['macd'][self.index]

    @property
    def macd_signal(self):
        return self._df['macd_signal'][self.index]

    @abstractmethod
    def handle_data(self, data):
        pass

    def buy(self):
        price = self._df['price'][self.index]
        self._cash -= price * self._TRANSACTION_FEE
        self._btc = self._cash / price
        self._cash = 0
        self.transactions.append(transaction(action.buy, self._df['time'][self.index]))

    def sell(self):
        price = self._df['price'][self.index]
        self._btc -= self._TRANSACTION_FEE
        self._cash = self._btc * price
        self._btc = 0
        self.transactions.append(transaction(action.sell, self._df['time'][self.index]))

    def hold(self):
        self.transactions.append(transaction(action.hold, self._df['time'][self.index]))

    def start(self):
        for i in range(len(self._df)):
            self.index = i
            self.handle_data(self.index)
            self.cash_history.append(self._cash + (self._btc * self.btc_price))
        return self.cash_history, self.transactions


class BTCStrategy(TestEnv):

    # handle data has it come in
    def handle_data(self, i):
        if self.rsi <= 20 and self.can_buy and abs(self.macd - self.macd_signal) < 0.2:
            self.buy()
        elif self.rsi >= 50 and self.can_sell and abs(self.macd - self.macd_signal) < 0.2:
            self.sell()
        else:
            self.hold()


def plot_cash(cash):
    plt.plot(cash)
    plt.title('Cash')
    plt.ylabel('$$$')
    plt.xlabel('Time')
    plt.show()


df = pd.read_feather('data/training_data.feather')[:100000]
strategy = BTCStrategy(df, 1000)
hist, transactions = strategy.start()
plot_cash(hist)
print(transactions)
