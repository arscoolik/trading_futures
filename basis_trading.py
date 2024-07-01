import argparse

from Logger import get_logger

from BinanceArb import BinanceArbBot
from detect_spread import *

multiplier = {
    'BTC': 100,  # 1 contract = 100USD
    'EOS': 10,  # 1 contract = 10USD
    'DOT': 10,
    'ETH': 10,
    'LTC': 10,
    'EOS': 10,
    'XRP': 10,
    'FIL': 10,
    'ADA': 10,
    'BCH': 10,
    'LINK': 10
}  


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', default='', help='exchange')
    parser.add_argument('--coin', type=str, default='XRP'.upper())
    parser.add_argument('--future_date', type=str, default='221230', help='expiration date')
    parser.add_argument('--coin_precision', type=int, default=4, help="price precision") # DEFAULT% BITCOIN
    parser.add_argument('--slippage', type=float, default=0.00025, help="proportion of coin price")
    parser.add_argument('--multiplier', type=dict, default=multiplier, help='expiration date')

    # commissionRate
    # BTC-related: 0; 25% off using BNB; BUSD pairs maker: 0
    parser.add_argument('--spot_fee_rate', type=float, default=0)
    parser.add_argument('--contract_fee_rate', type=float, default=0, help="maker: 0.01%, taker: 0.05%")
    parser.add_argument('--max_trial', type=int, default=5, help="number of trials for connection")
    parser.add_argument('--api_key', type=str, default='', help="api-key")
    parser.add_argument('--secret_key', type=str, default='', help="secret-key")
    parser.add_argument('--futures_leverage', type=int, default=5, help="futures_leverage")
    parser.add_argument('--debug_enabled', type=bool, default=False, help="debug mode is enabled")
    return parser

def get_trading_coin(exchange, LOGGER):
    spread_info , _= exchange.get_spread_info(logger=LOGGER)
    return spread_info[1][:-5]


if __name__ == '__main__':

    # ***open positions***
    position_parser = init_argparse()
    position_parser.add_argument('--amount', type=int, default=20, help="spot trading amount for one iteration")
    position_parser.add_argument('--num_maximum', type=int, default=3, help="maximum execution numbers")
    position_parser.add_argument('-f', '--threshold', type=float, default=0.01, help="opening threshold")
    position_parser.add_argument('--negative_threshold', type=float, default=-0.05, help="negative threshold")
    position_parser.add_argument('--required_iterations', type=int, default=5, help="number of success required iterations")
    args = position_parser.parse_args()

    trading_bot = BinanceArbBot(**vars(args))

    exchange = BA(trading_bot.secret_key, trading_bot.api_key)
    LOGGER = get_logger("Spread Detection")
    
    spread, tm, k_parameter = 0, 0, 6
    while True:
        # trading_bot.coin = get_trading_coin(exchange, LOGGER)
        trading_bot.update_symbols()
        trading_bot.open_position(spread, tm, k_parameter)
        spread, tm, k_parameter = trading_bot.close_position()
        print(f"args.debug_enabled={args.debug_enabled}")
        if args.debug_enabled == True: # only 1 circle in debug mode
            exit()


