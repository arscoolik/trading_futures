import ccxt
import time
import math
import traceback
from Logger import get_logger
import numpy as np
from collections import deque


class BinanceArbBot:
    def __init__(self, exchange: ccxt.binance, coin: str, future_date: str,
                 coin_precision: float, slippage: float, spot_fee_rate: float,
                 contract_fee_rate: float, multiplier: dict,
                 amount: float, num_maximum: int, threshold: float,
                 negative_threshold: float, required_iterations: int,
                 max_trial: int, futures_leverage: int, api_key: str,
                 secret_key: str, debug_enabled: bool):
        self.debug_enabled = debug_enabled
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'timeout': 3000,
            'rateLimit': 10,
            'verbose': False,
            'enableRateLimit': True})

        self.exchange_futures = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'timeout': 3000,
            'rateLimit': 10,
            'verbose': False,
            'enableRateLimit': True
        })
        self.exchange_futures.options = {'defaultType': 'delivery',
                                         'adjustForTimeDifference': True
                                         }
        self.secret_key = secret_key
        self.api_key = api_key
        self.coin = coin
        self.future_date = future_date
        self.coin_precision = coin_precision
        self.slippage = slippage
        self.spot_fee_rate = spot_fee_rate
        self.contract_fee_rate = contract_fee_rate
        self.multiplier = multiplier
        self.initial_amount = amount
        self.amount = amount
        self.num_maximum = num_maximum
        self.threshold = threshold
        self.negative_threshold = negative_threshold
        self.required_iterations = required_iterations
        self.max_trial = max_trial
        self.futures_leverage = futures_leverage
        self.period = 14
        self.last_deals = deque(maxlen=self.period)
        self.logger = get_logger("Basis-Trading Starts")
        self.spot_symbol = {'type1': coin + 'FDUSD', 'type2': coin + '/FDUSD'}
        self.future_symbol = {'type1': coin + 'USD_' + future_date}

        self.state = {}


    def calculate_rsi(self, prices):
        period = self.period
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]  # на один элемент меньше

            if delta > 0:
                up_val = delta
                down_val = 0.
            else:
                up_val = 0.
                down_val = -delta

            up = (up * (period - 1) + up_val) / period
            down = (down * (period - 1) + down_val) / period

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi[0]

    def update_symbols(self):
        self.spot_symbol = {'type1': self.coin +
                            'FDUSD', 'type2': self.coin + '/FDUSD'}
        self.future_symbol = {'type1': self.coin + 'USD_' + self.future_date}

    def retry_wrapper(
            self,
            func,
            params=dict(),
            act_name='',
            sleep_seconds=5,
            is_exit=True):

        for _ in range(self.max_trial):
            try:
                # NOTE: reset the local timestamp when requesting
                # again，otherwise requests may be declined
                if isinstance(
                        params, dict) and 'timestamp' in list(
                        params.keys()):
                    params['timestamp'] = int(time.time()) * 1000
                result = func(**params)
                return result
            except Exception as e:
                self.logger.warning(f'Got an exception: {e}')
                self.logger.warning(
                    f"{act_name} ({self.exchange.id}) FAIL | Retry after {sleep_seconds} seconds...")
                self.logger.warning('Retrying function parameters:', params)
                self.logger.warning(traceback.format_exc())
                time.sleep(sleep_seconds)
        else:
            self.logger.critical(
                f"{act_name} FAIL too many times... Arbitrage STOPs!")
            if is_exit:
                exit()

    def binance_spot_place_order(
            self,
            symbol: str,
            direction: str,
            price: float,
            amount: float):

        # Проверка направления ордера и выставление маркет-лимитного ордера с
        # указанием цены или без нее
        if direction == 'long':
            # Проверяем, нужно ли указывать цену для маркет-лимитного ордера
            if price:
                order_info = self.exchange.create_limit_buy_order(
                    symbol, amount, price)
            else:
                order_info = self.exchange.create_market_buy_order(
                    symbol, amount)
        elif direction == 'short':
            if price:
                order_info = self.exchange.create_limit_sell_order(
                    symbol, amount, price)
            else:
                order_info = self.exchange.create_market_sell_order(
                    symbol, amount)
        else:
            raise ValueError(
                'Parameter `direction` supports `long` or `short` only')

        self.logger.debug(
            f'Spot orders ({self.exchange.id}) SUCCESS: {direction} > {symbol} > {amount} > {price}')
        self.logger.debug(f'Order info: {str(order_info)}')

        return order_info

    def get_futures_balance(self):
        futures_balance_info = self.exchange_futures.fetch_balance()
        self.logger.debug(f"getting {self.coin} futures balance")
        return float(futures_balance_info[self.coin]['free'])

    def binance_future_place_order(
            self,
            symbol: str,
            direction: str,
            amount: int,
            type: str):

        if direction == 'open_short':
            side = 'SELL'
        elif direction == 'close_short':
            side = 'BUY'
        else:
            raise NotImplementedError(
                'Parameter `direction` only supports `open_short` and `close_short` currently')

        if type == 'MARKET':
            params = {
                'symbol': symbol,
                'side': side,
                'type': type,
                'quantity': amount,
                'timestamp': int(time.time() * 1000),
            }
        # elif type == 'LIMIT':
        #     params = {
        #         'symbol': symbol,
        #         'side': side,
        #         'type': type,
        #         'price': price * (1 + self.slippage),
        #         'timeInForce': 'GTC',
        #         # 'reduce': reduce,
        #         'quantity': amount,
        #         'timestamp': int(time.time() * 1000),
        #     }

        params['timestamp'] = int(time.time() * 1000)
        order_info = self.exchange.dapiPrivatePostOrder(params)
        self.logger.debug(
            f'({self.exchange.id}) Future orders SUCCESS: {direction} > {symbol} > {amount}')
        self.logger.debug(f'Order info: {str(order_info)}')

        return order_info

    def binance_account_transfer(
            self,
            currency: str,
            amount,
            from_account='spot',
            to_account='coin-margin'):
        """
        POST /sapi/v1/asset/transfer (HMAC SHA256)
        """

        if from_account == 'spot' and to_account == 'coin-margin':
            transfer_type = 'MAIN_CMFUTURE'
        elif from_account == 'coin-margin' and to_account == 'spot':
            transfer_type = 'CMFUTURE_MAIN'
        else:
            raise ValueError(
                'Cannot recognize parameters for User Universal Transfer')

        params = {
            'type': transfer_type,
            'asset': currency,
            'amount': amount,
        }
        params['timestamp'] = int(time.time() * 1000)
        transfer_info = self.exchange.sapiPostAssetTransfer(params=params)
        self.logger.debug(
            f"({self.exchange.id}) Transfer SUCCESS: {from_account} --> {to_account} > amount: {amount}")
        self.logger.debug(f'Transfer info: {str(transfer_info)}')

        return transfer_info

    def open_position(self):
        execute_num = 0

        while True:
            spot_ask1 = float(self.exchange.publicGetTickerBookTicker(
                params={'symbol': self.spot_symbol['type1']})['bidPrice'])
            coin_bid1 = float(self.exchange.dapiPublicGetTickerBookTicker(
                params={'symbol': self.future_symbol['type1']})[0]['askPrice'])
            
            while len(self.last_deals) < self.period:
                self.last_deals.append(coin_bid1)
                coin_bid1 = float(self.exchange.dapiPublicGetTickerBookTicker(
                params={'symbol': self.future_symbol['type1']})[0]['askPrice'])
                time.sleep(10)
                self.logger.info(self.last_deals)

            rsi = self.calculate_rsi(self.last_deals)
            while rsi < 70:
                self.logger.info(f"Waiting for the perfect moment. Current rsi is {rsi}\n{self.last_deals}")
                self.last_deals.popleft()
                coin_bid1 = float(self.exchange.dapiPublicGetTickerBookTicker(
                params={'symbol': self.future_symbol['type1']})[0]['askPrice'])
                self.last_deals.append(coin_bid1)
                rsi = self.calculate_rsi(list(self.last_deals))
                time.sleep(120)

                
            r = coin_bid1 / spot_ask1 - 1

            if True:
                self.logger.info(
                    'Spot %.4f < COIN-M %.4f -> Price Difference: %.4f%%' %
                    (float(spot_ask1), float(coin_bid1), r * 100))
                self.logger.debug('!!! >>> Starting arbitrage...')

                # Размещение ордера на споте
                price = float(spot_ask1) * (1 - self.slippage)
                spot_amount = math.floor(self.amount / price)
                params = {
                    'symbol': self.spot_symbol['type2'],
                    'direction': 'long',
                    'price': price,
                    'amount': spot_amount,
                }
                spot_order_info = self.retry_wrapper(
                    func=self.binance_spot_place_order,
                    params=params,
                    act_name='Long spot orders')
                start_time = time.time()
                # Проверка успешности размещения ордера на споте
                while (
                    self.exchange.fetch_balance()[
                        self.coin]['free'] < spot_amount):
                    self.logger.warning(
                        f'Insufficient {self.coin} balance on spot account, required: ({spot_amount})')
                    time.sleep(3)
                    if (time.time() - start_time) / 60 > 5:
                        self.logger.error(
                            'Failed to place spot order in 5 mins, cancelling...')
                        cancel_order_params = {
                            'orderId': spot_order_info['info']['orderId'],
                            'symbol': spot_order_info['info']['symbol'],
                        }

                        self.retry_wrapper(
                            func=self.cancel_order,
                            params=cancel_order_params,
                            act_name="cancel long spot order")
                        spot_ask1 = float(self.exchange.publicGetTickerBookTicker(
                            params={'symbol': self.spot_symbol['type1']})['bidPrice'])
                        spot_order_params = {
                            'symbol': self.spot_symbol['type2'],
                            'direction': 'long',
                            'price': float(spot_ask1) * (1 - self.slippage),
                            'amount': spot_amount,
                        }
                        start_time = time.time()
                        spot_order_info = self.retry_wrapper(
                            func=self.binance_spot_place_order,
                            params=spot_order_params,
                            act_name='Long spot orders')

                # Если ордер на споте успешно размещен, переводим средства на
                # coin-margin

                # balance = self.exchange.fetch_balance()
                # num = balance[self.coin]['free']
                # self.logger.debug(f'Amount to be transferred > {num}')

                params = {
                    'currency': self.coin,
                    'amount': spot_amount,
                    'from_account': 'spot',
                    'to_account': 'coin-margin',
                }
                transfer_info = self.retry_wrapper(
                    func=self.binance_account_transfer,
                    params=params,
                    act_name='Transfer (SPOT --> COIN-M)')

                # Проверка успешности перевода на coin-margin
                if not transfer_info:
                    self.logger.warning(
                        'Transfer to coin-margin failed, retrying...')
                    continue

                # Если перевод на coin-margin успешно выполнен, размещаем ордер на coin-margin
                # price = float(coin_bid1) * (1 - self.slippage)
                coin_bid1 = float(self.exchange.dapiPublicGetTickerBookTicker(
                    params={'symbol': self.future_symbol['type1']})[0]['askPrice'])
                futures_contract_num = math.floor(
                    spot_amount * coin_bid1 * self.futures_leverage / self.multiplier[self.coin]) - 1
                params = {
                    'symbol': self.future_symbol['type1'],
                    'direction': 'open_short',
                    'amount': futures_contract_num,
                    'type': 'MARKET'
                }
                self.logger.info(f"future place order parameters: {params}")
                future_order_info = self.retry_wrapper(
                    func=self.binance_future_place_order,
                    params=params,
                    act_name='Short coin-margin orders')
                self.logger.info(future_order_info)
                r = coin_bid1 / spot_ask1 - 1
                self.logger.info(f"OPENING FUTURES SHORT: coin_bid1 = {coin_bid1}, self.multiplier[self.coin] = {self.multiplier[self.coin]}")
                if future_order_info:
                    self.state = {
                        'quantity': futures_contract_num,
                        'open_spread': r,
                        'orderId': future_order_info['orderId'],
                        'initial_future_price': coin_bid1,
                        'initial_spot_price': spot_ask1
                    }
                    execute_num += 1
                    self.logger.info(
                        f"Number of opening executions: {execute_num}")

                if execute_num >= self.num_maximum:
                    self.logger.info(
                        'Maximum execution number reached >>> Position opening stops.')
                    break
            time.sleep(5)  # Не перегружаем биржу

    def close_position_utils(self):
        """close positions for basis trading"""
        balance = self.exchange.fetch_balance()
        num = balance['FDUSD']['free']
        self.logger.info(f'Amount of USD in spot account：{num}')

        now_execute_num = 0
        success_spread_difference_num = 0

        initial_future_price = self.state.get('initial_future_price')
        initial_spot_price = self.state.get('initial_spot_price')

        while True:
            self.logger.info(
                f'Currently trading {self.amount}$ of {self.coin}')
            spot_sell_price = float(self.exchange.publicGetTickerBookTicker(
                params={'symbol': self.spot_symbol['type1']})['bidPrice'])
            future_buy_price = float(self.exchange.dapiPublicGetTickerBookTicker(
                params={'symbol': self.future_symbol['type1']})[0]['askPrice'])

            self.logger.info(
                f'Initial spot price: {initial_spot_price}\nCurrent spot price: {spot_sell_price}')
            self.logger.info(
                f'Initial future price: {initial_future_price}\nCurrent future price: {future_buy_price}')
            calculated_profit = self.calculate_profit(
                initial_spot_price,
                spot_sell_price,
                initial_future_price,
                future_buy_price,
                0.0005,
                math.floor(self.amount / initial_spot_price) * initial_spot_price,
                self.futures_leverage)
            self.logger.info(
                f'Calculated possible profit: {calculated_profit * self.amount}$')

            if not self.debug_enabled:
                if calculated_profit < self.negative_threshold or calculated_profit > self.threshold:
                    success_spread_difference_num += 1
                else:
                    success_spread_difference_num = 0
                    continue
                if success_spread_difference_num < self.required_iterations:
                    continue

            self.logger.debug(
                'Spread difference GREATER than threshold >>> Stopping arbitrage...')

            futures_contract_num = self.state.get('quantity')
            price = future_buy_price
            price = round(price, self.coin_precision)

            params = {
                'symbol': self.future_symbol['type1'],
                'direction': 'close_short',
                'amount': futures_contract_num,
                'type': 'MARKET'
            }

            self.logger.info(f"future place order: {params}")
            self.retry_wrapper(
                func=self.binance_future_place_order,
                params=params,
                act_name='Close short coin-margin orders')
            time.sleep(2)
            futures_balance = self.get_futures_balance()
            params = {
                'currency': self.coin,
                'amount': futures_balance,
                'from_account': 'coin-margin',
                'to_account': 'spot',
            }
            self.retry_wrapper(
                func=self.binance_account_transfer,
                params=params,
                act_name='Transfer (COIN-M --> SPOT)')  # TODO: transfer all coins

            balance = self.exchange.fetch_balance()
            num = balance[self.coin]['free']
            self.logger.info(
                f'Amount of {self.coin} in coin-margin account：{num}')

            if num < self.amount:
                self.logger.error(
                    'Please ensure the coin-margin remaining balance is enough!')

            price = spot_sell_price * (1 + self.slippage)
            params = {
                'symbol': self.spot_symbol['type2'],
                'direction': 'short',
                'price': price,
                'amount': num,
            }
            self.retry_wrapper(
                func=self.binance_spot_place_order,
                params=params,
                act_name='Short spot orders')

            start_time = time.time()
            required_coin_balance = 1 / \
                self.multiplier[self.coin] * futures_balance
            while self.exchange.fetch_balance()['FDUSD']['free'] < price * num:
                self.logger.warning(
                    f'Too many {self.coin} on spot account, required: {required_coin_balance}')
                time.sleep(3)
                if (time.time() - start_time) / 60 > 5:  # more than 5 mins
                    self.logger.error(
                        f'Already {(time.time() - start_time) // 60} minutes has passed trying to close SPOT')

            profit = (price * num - self.amount) / self.amount
            profit_usd = profit * self.amount
            # self.logger.info(
            #     f"Circle finished:\nProfit: {profit * 100}% | {profit_usd}$\n")
            with open("balance.csv", "a") as f:
                f.write(
                    f"{(self.exchange.fetch_balance())['FDUSD']['free']},{str(time.time())}\n")
            
            if profit_usd > 0:
                self.logger.info(f"=) Trading circle ended in plus: Profit: {profit * 100}% | {profit_usd}$")
                next_circle_amount = self.amount + profit_usd / 2
                self.logger.info(f"Next circle money amount: {next_circle_amount}$")

            if profit_usd < 0:
                self.logger.info(f"=( Trading circle ended in minus: Loss: {profit * 100}% | {profit_usd}$")
                next_circle_amount = max(self.initial_amount, self.amount + profit_usd)
                self.logger.warning('Closed badly, so now im sleeping for 4 hours')
                time.sleep(14400)

            # now_execute_num = now_execute_num + 1

            # self.logger.info(f"Number of closing executions: {now_execute_num}")

            

            if now_execute_num >= self.num_maximum:
                self.logger.info(
                    'Maximum execution number reached >>> Position closing stops.')
            break



    def close_position(self):
        while True:
            try:
                self.close_position_utils()
            except Exception as e:
                self.logger.critical(
                    f'Closing positions FAILED with Exception: {e}>>> Retrying...')
                self.logger.warning(traceback.format_exc())
                time.sleep(2)

    def cancel_order(self, orderId: str, symbol: str):
        return self.exchange.cancel_order(orderId, symbol)

    def calculate_profit(
            self,
            initial_spot_price,
            final_spot_price,
            initial_future_price,
            final_future_price,
            futures_fee,
            amount,
            leverage):
        spot_coin_num = amount / initial_spot_price
        futures_coin_num = math.floor(
            spot_coin_num * leverage * initial_future_price / 10) * 10 / initial_future_price * (
            1 - futures_fee)
        futures_coin_num_new = ((initial_future_price - final_future_price) /
                                initial_future_price + 1) * futures_coin_num * (1 - futures_fee)
        spot_coin_num_new = math.floor(
            spot_coin_num + futures_coin_num_new - futures_coin_num)
        return (spot_coin_num_new * final_spot_price - amount) / amount

    def calculate_k_hour_parameter(self, profit: float):
        return 24
