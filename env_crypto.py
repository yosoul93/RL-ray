
import random
import pyfolio
import numpy as np
import pandas as pd
from copy import deepcopy
from talib.abstract import *
from gym.spaces import Box, Discrete

class observation_space:
    def __init__(self, a, b):
        self.shape = (a, b)

class action_space:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)

class CryptoEnv:  # custom env
    def __init__(self, initial_account=1e4, data=None, lag=None,
                ptc=1e-3, mode='train', features=None):
        self.initial_account = initial_account
        self.data = data
        self.ptc = ptc
        self.lag = lag
        self.mode = mode
        self.features = features
        self.n_features = len(features)
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lag, self.n_features),
            dtype=np.float32
        )
        # reset
        self.bar = self.lag
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.units = 0.0 
        
        self._load_data()
        self._prepare_data()

        # self.day_price = self._get_current_price()

        # self.total_asset = self.account + self.day_price * self.units
        # self.episode_return = 0.0  
        # self.gamma_return = 0.0
        

        '''env information'''
        # self.env_name = 'BitcoinEnv4'
        # self.state_dim = (3,)
        # self.action_dim = 1
        self.target_return = 10
        self.max_step = self.data[self.lag:].shape[0]

    def _load_data(self):
        split = int(self.data.shape[0]*0.85)
        data_train = self.data[:split]
        data_test = self.data[split:]
        
        if self.mode == 'train':
            self.data = data_train
        elif self.mode == 'test':
            self.data = data_test
        else:
            raise ValueError('Invalid Mode!')

    def _prepare_data(self):
        self.data["account"] = 0
        self.data["units"] = 0
        self.data['r'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['m'] = self.data['r'].rolling(30).mean()
        self.data['v'] = self.data['r'].rolling(30).std()
        self.data['min'] = self.data['close'].rolling(14).min()
        self.data['max'] = self.data['close'].rolling(14).max()
        self.data['mami'] = self.data['max'] - self.data['min']
        self.data['mac'] = abs(self.data['max'] - self.data['close'].shift(1))
        self.data['mic'] = abs(self.data['min'] - self.data['close'].shift(1))
        self.data['atr'] = np.maximum(self.data['mami'], self.data['mac'])
        self.data['atr'] = np.maximum(self.data['atr'], self.data['mic'])
        self.data['atr%'] = self.data['atr'] / self.data['close']
        self.data['avg_price'] = AVGPRICE(self.data['open'], self.data['high'], self.data['low'], self.data['close'])
        self.data['med_price'] = MEDPRICE(self.data['high'], self.data['low'])
        self.data['typ_price'] = TYPPRICE(self.data['high'], self.data['low'], self.data['close'])
        self.data['wcl_price'] = WCLPRICE(self.data['high'], self.data['low'], self.data['close'])
        self.data['upperband'], self.data['middleband'], self.data['lowerband'] = BBANDS(self.data['close'], timeperiod=5, matype=0)
        self.data['sma_7'] = SMA(self.data['close'], timeperiod=7)
        self.data['sma_25'] = SMA(self.data['close'], timeperiod=25)
        self.data['sma_99'] = SMA(self.data['close'], timeperiod=99)
        self.data['ema_7'] = EMA(self.data['close'], timeperiod=7)
        self.data['ema_25'] = EMA(self.data['close'], timeperiod=25)
        self.data['ema_99'] = EMA(self.data['close'], timeperiod=99) 
        self.data['macd'], self.data['macd_signal'], self.data['macd_hist'] = MACD(self.data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.data['rsi'] = RSI(self.data['close'], timeperiod=30)
        self.data['cci'] = CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=30)
        self.data['dx'] = DX(self.data['high'], self.data['low'], self.data['close'], timeperiod=30)

        self.data.dropna(inplace=True)
        # if self.mu is None:
        #     self.mu = self.data.mean()
        #     self.std = self.data.std()
        # self.data_ = (self.data - self.mu) / self.std
        # self.data_ = (self.data - self.data.mean()) / self.data.std()
        # self.data_ = self.data * 2 ** -8
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data['d'] = self.data['d'].astype(int)
        

    def _get_current_price(self):
        return self.data_['close'].iat[self.bar]

    def _set_account_and_unit_in_data(self):
        self.data_['account'].iat[self.bar] = self.account
        self.data_['units'].iat[self.bar] = self.units

    def _get_state(self):
        self._set_account_and_unit_in_data()
        return self.data_[self.features].iloc[self.bar - self.lag:self.bar]

    def get_state(self, bar, account, unit):
        self.data_['account'].iat[self.bar] = account
        self.data_['units'].iat[self.bar] = unit
        return self.data[self.features].iloc[bar - self.lag:bar]


    def reset(self) -> np.ndarray:
        self.bar = self.lag
        self.data_ = deepcopy(self.data)
        self.day_price = self.init_price = self._get_current_price()
        self.initial_account__reset = self.initial_account  # reset()
        self.account = self.initial_account__reset
        self.units = 0.0
        self.total_asset = self.account + self.day_price * self.units
        
        state = self._get_state().values.astype(np.float64) * 2 ** -15
        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
        # print("action:", action)
        stock_action = 1 if action == 1 else -1
        """buy or sell stock"""
        adj = self._get_current_price()

        if stock_action == 1: # buy
            if self.units <= 0.0:
                delta_uniit = (self.total_asset / adj) - self.units
                self.account -= adj * delta_uniit * (1 + self.ptc)
                self.units += delta_uniit
                # print("BUY:", self.account, self.units)
        elif stock_action == -1: # sell
            if self.units >= 0.0:
                delta_uniit = (self.total_asset / adj) + self.units
                self.account += adj * delta_uniit * (1 - self.ptc)
                self.units -= delta_uniit
                # print("SELL:", self.account, self.units)
            
        """update bar"""
        self.bar += 1
        done = (self.bar + 1) == self.max_step 

        state = self._get_state().values.astype(np.float64) * 2 ** -15

        next_adj = self._get_current_price()
        next_total_asset = self.account + next_adj*self.units
        reward = (next_total_asset - self.total_asset) * 2 ** -15  
        self.total_asset = next_total_asset

        # self.gamma_return = self.gamma_return * self.gamma + reward 
        if done:
            # reward += self.gamma_return
            # self.gamma_return = 0.0  
            self.episode_return = next_total_asset / self.initial_account 

            print("episode_return:", round(self.episode_return, 4), "buy_hold:", round(next_adj/self.init_price, 4))
        info = {}
        return state, reward, done, info
    
    def draw_cumulative_return(self, model) -> list:
        state = self.reset()
        state = np.reshape(state, [1, self.lag, self.n_features])
        episode_returns = list()
        episode_returns.append(1)
        buy_hold_returns = list()
        for i in range(self.max_step):
            
            adj = self._get_current_price()
            if i == 0:
                init_price = adj
            
            action = np.argmax(model.predict(state)[0, 0])
            next_state, reward, done, info = self.step(action)
            state = np.reshape(next_state, [1, self.lag, self.n_features])
                
            total_asset = self.account + (adj * self.units)
            episode_returns.append(total_asset / self.initial_account)
            buy_hold_returns.append(adj/init_price)
            if done:
                break

        import matplotlib.pyplot as plt
        plt.figure(figsize=(17, 6))
        plt.plot(episode_returns, label='Agent Return')
        plt.plot(buy_hold_returns, label = 'Buy Hold return')
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.legend()
        plt.show()

