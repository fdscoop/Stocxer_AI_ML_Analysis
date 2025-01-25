from flask import Flask, request, jsonify
import requests
import json
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, List
import numpy as np
from typing import Dict, Any
from scipy.stats import norm 
import math
from typing import Dict, List, Any, Tuple
import os




class OptionsGreeksCalculator:
    """Calculate and analyze options Greeks for index options trading"""
    def __init__(self, risk_free_rate: float = 0.07):  # Using 7% as typical Indian risk-free rate
        self.risk_free_rate = risk_free_rate

    def calculate_greeks(self, 
                        spot_price: float,
                        strike_price: float,
                        time_to_expiry: float,  # in years
                        volatility: float,
                        option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Args:
            spot_price: Current index price
            strike_price: Option strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility (as decimal)
            option_type: 'call' or 'put'
            
        Returns:
            Dict containing all Greeks values
        """
        if time_to_expiry <= 0:
            return {
                'delta': 1.0 if option_type == 'call' else -1.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }

        # Calculate d1 and d2 for Black-Scholes
        d1 = (np.log(spot_price / strike_price) + 
              (self.risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        # Calculate option price and Greeks
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / 
                    (2 * np.sqrt(time_to_expiry)) - 
                    self.risk_free_rate * strike_price * 
                    np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        else:  # put
            delta = -norm.cdf(-d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / 
                    (2 * np.sqrt(time_to_expiry)) + 
                    self.risk_free_rate * strike_price * 
                    np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2))

        # Common Greeks for both call and put
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        vega = spot_price * np.sqrt(time_to_expiry) * norm.pdf(d1) / 100  # Divided by 100 for percentage
        rho = (strike_price * time_to_expiry * 
               np.exp(-self.risk_free_rate * time_to_expiry) * 
               (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))) / 100

        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

class IndexOptionsAnalyzer:
    """Analyze index options considering futures data and Greek values"""
    def __init__(self, greeks_calculator: OptionsGreeksCalculator):
        self.greeks_calculator = greeks_calculator

    def select_optimal_options(self,
                             current_price: float,
                             options_chain: Dict[str, List[Dict]],
                             futures_data: Dict[str, Any],
                             vix: float) -> Dict[str, List[Dict]]:
        """Select optimal options with fallback to theoretical options"""
        if not options_chain['calls'] and not options_chain['puts']:
            # Generate theoretical options
            theoretical_options = self._generate_theoretical_options(current_price, vix)
            options_chain = theoretical_options

        # Select strikes around ATM
        atm_strike = self._find_atm_strike(current_price, options_chain)
        strikes_to_analyze = {
            'calls': self._get_nearby_strikes(atm_strike, options_chain['calls'], 3),
            'puts': self._get_nearby_strikes(atm_strike, options_chain['puts'], 3)
        }

        selected_options = {'calls': [], 'puts': []}
        volatility = vix / 100

        for option_type in ['calls', 'puts']:
            for option in strikes_to_analyze[option_type]:
                time_to_expiry = self._calculate_time_to_expiry(option['expiry'])
                greeks = self.greeks_calculator.calculate_greeks(
                    current_price,
                    option['strike_price'],
                    time_to_expiry,
                    volatility,
                    'call' if option_type == 'calls' else 'put'
                )

                enhanced_option = {
                    **option,
                    'greeks': greeks,
                    'liquidity_score': self._calculate_liquidity_score(option, futures_data),
                    'entry_zones': self._calculate_entry_zones(option, greeks, current_price, vix)
                }
                selected_options[option_type].append(enhanced_option)

        return selected_options

    def _generate_theoretical_options(self, current_price: float, vix: float) -> Dict[str, List[Dict]]:
        """Generate theoretical options around current price"""
        strikes = []
        base_strike = round(current_price / 50) * 50  # Round to nearest 50
        
        # Generate strikes ±5% around current price
        for i in range(-5, 6):
            strikes.append(base_strike + (i * 50))

        theoretical_options = {'calls': [], 'puts': []}
        expiry_dates = ['2025-02-01', '2025-02-07', '2025-02-14']  # Weekly expiries
        
        for strike in strikes:
            for expiry in expiry_dates:
                time_to_expiry = self._calculate_time_to_expiry(expiry)
                volatility = vix / 100
                
                # Calculate theoretical price using Black-Scholes
                call_price = self._calculate_theoretical_price(
                    current_price, strike, time_to_expiry, volatility, 'call'
                )
                put_price = self._calculate_theoretical_price(
                    current_price, strike, time_to_expiry, volatility, 'put'
                )

                # Add call option
                theoretical_options['calls'].append({
                    'strike_price': strike,
                    'expiry': expiry,
                    'ltp': call_price,
                    'volume': futures_data.get('volume', 0) * 0.1,  # 10% of futures volume
                    'openInterest': futures_data.get('oi', 0) * 0.1
                })

                # Add put option
                theoretical_options['puts'].append({
                    'strike_price': strike,
                    'expiry': expiry,
                    'ltp': put_price,
                    'volume': futures_data.get('volume', 0) * 0.1,
                    'openInterest': futures_data.get('oi', 0) * 0.1
                })

        return theoretical_options

    def _calculate_theoretical_price(self,
                                  spot: float,
                                  strike: float,
                                  time: float,
                                  vol: float,
                                  option_type: str) -> float:
        """Calculate theoretical option price using Black-Scholes"""
        greeks = self.greeks_calculator.calculate_greeks(spot, strike, time, vol, option_type)
        if option_type == 'call':
            return max(spot - strike, 0) + greeks['theta'] * time
        else:
            return max(strike - spot, 0) + greeks['theta'] * time
        

    def _calculate_liquidity_score(self,
                                 option: Dict[str, Any],
                                 futures_data: Dict[str, Any]) -> float:
        """Calculate liquidity score using futures volume data"""
        futures_volume = futures_data.get('volume', 0)
        option_volume = option.get('volume', 0)
        
        # Use futures volume as baseline if option volume is zero
        if option_volume == 0:
            option_volume = futures_volume * 0.1  # Assume 10% of futures volume
            
        spread = abs(option.get('ask', 0) - option.get('bid', 0))
        normalized_spread = spread / option.get('ltp', 1)
        
        # Score components
        volume_score = min(option_volume / futures_volume, 1)
        spread_score = 1 - min(normalized_spread, 1)
        
        return (volume_score * 0.7 + spread_score * 0.3)

    def _calculate_entry_zones(self,
                             option: Dict[str, Any],
                             greeks: Dict[str, float],
                             current_price: float,
                             vix: float) -> Dict[str, float]:
        """Calculate entry and exit zones based on Greeks and volatility"""
        ltp = option.get('ltp', 0)
        delta = abs(greeks['delta'])
        
        # Adjust ranges based on VIX
        volatility_factor = vix / 20  # Normalize around VIX of 20
        
        # Entry zones with tighter ranges for index options
        entry_low = ltp * (1 - 0.03 * volatility_factor * (1 - delta))
        entry_high = ltp * (1 + 0.02 * volatility_factor * (1 - delta))
        
        # Exit zones
        stop_loss = ltp * (1 - 0.10 * volatility_factor)
        target = ltp * (1 + 0.15 * volatility_factor)
        
        return {
            'entry_zone': {
                'low': entry_low,
                'high': entry_high
            },
            'exit_zone': {
                'stop_loss': stop_loss,
                'target': target
            }
        }

    # First, let's fix the IndexOptionsAnalyzer class's _find_atm_strike method
    def _find_atm_strike(self,
                        current_price: float,
                        options_chain: Dict[str, List[Dict]]) -> float:
        """
        Find the At-The-Money strike price safely, handling empty data cases
        
        Args:
            current_price: Current index price
            options_chain: Dictionary containing calls and puts lists
            
        Returns:
            float: ATM strike price, or current price if no valid strikes found
        """
        try:
            all_strikes = []
            for option_type in ['calls', 'puts']:
                # Safely extract strike prices, handling potential missing data
                strikes = options_chain.get(option_type, [])
                if strikes:
                    strikes = [opt.get('strike_price', 0) for opt in strikes if opt.get('strike_price')]
                    all_strikes.extend(strikes)
            
            # If we have valid strikes, find the closest to current price
            if all_strikes:
                return min(all_strikes, key=lambda x: abs(x - current_price))
            else:
                # If no valid strikes found, return the current price rounded to nearest 50
                return round(current_price / 50) * 50
                
        except Exception as e:
            logger.error(f"Error finding ATM strike: {e}")
            return current_price  # Fallback to current price if anything goes wrong
    
    def _get_nearby_strikes(self,
                          atm_strike: float,
                          options: List[Dict],
                          count: int) -> List[Dict]:
        """Get nearby strike prices centered around ATM"""
        sorted_options = sorted(options, key=lambda x: abs(x['strike_price'] - atm_strike))
        return sorted_options[:count]

    def _calculate_time_to_expiry(self, expiry_date: str) -> float:
        """Calculate time to expiry in years"""
        from datetime import datetime
        current_date = datetime.now()
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
        days_to_expiry = (expiry - current_date).days
        return max(days_to_expiry / 365, 0.00001)  # Avoid divide by zero


class OptionsStrategyGenerator:
    """Generate trading strategies for index options considering Greeks and expiry timing"""
    def __init__(self, vix_threshold: float = 20.0):
        self.vix_threshold = vix_threshold

    def generate_trading_strategy(self,
                                market_data: Dict[str, Any],
                                optimal_options: Dict[str, List[Dict]],
                                technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive options trading strategy based on market conditions"""
        # Extract market conditions
        vix = market_data.get('current_market', {}).get('vix', {}).get('ltp', 0)
        trend = technical_analysis.get('momentum_indicators', {}).get('trend_direction', 'Neutral')
        rsi = technical_analysis.get('momentum_indicators', {}).get('rsi', 50)
        
        is_high_volatility = vix > self.vix_threshold
        
        # Generate complete strategy
        primary_strategy = self._select_primary_strategy(
            trend=trend,
            rsi=rsi,
            vix=vix,
            optimal_options=optimal_options
        )
        
        hedge_strategy = self._generate_hedge_strategy(
            primary_strategy,
            optimal_options,
            is_high_volatility
        )
        
        position_sizes = self._calculate_position_sizes(
            optimal_options,
            primary_strategy['strategy_type'],
            vix
        )
        
        return {
            'primary_strategy': primary_strategy,
            'hedge_strategy': hedge_strategy,
            'position_sizing': position_sizes,
            'risk_parameters': self._generate_risk_parameters(vix),
            'execution_guidelines': self._generate_execution_guidelines(
                primary_strategy['strategy_type'],
                is_high_volatility
            )
        }

    def _select_primary_strategy(self,
                               trend: str,
                               rsi: float,
                               vix: float,
                               optimal_options: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Enhanced strategy selection with improved market condition handling"""
        try:
            strategies = []
            
            # Get ATM options
            atm_call = optimal_options.get('calls', [{}])[0] if optimal_options.get('calls') else {}
            atm_put = optimal_options.get('puts', [{}])[0] if optimal_options.get('puts') else {}
            
            # Strong Bearish Conditions
            if trend == 'Bearish' and rsi < 30:
                if vix > self.vix_threshold:
                    strategies.append({
                        'strategy_type': 'BEAR_PUT_SPREAD',
                        'primary_leg': atm_put,
                        'secondary_leg': optimal_options.get('puts', [{}])[-1] if optimal_options.get('puts') else {},
                        'rationale': 'Strong bearish trend with oversold RSI in high volatility',
                        'confidence': 'high'
                    })
                else:
                    strategies.append({
                        'strategy_type': 'LONG_PUT',
                        'primary_leg': atm_put,
                        'rationale': 'Strong bearish trend with oversold RSI in low volatility',
                        'confidence': 'high'
                    })
            
            # Strong Bullish Conditions
            elif trend == 'Bullish' and rsi > 70:
                if vix > self.vix_threshold:
                    strategies.append({
                        'strategy_type': 'BULL_CALL_SPREAD',
                        'primary_leg': atm_call,
                        'secondary_leg': optimal_options.get('calls', [{}])[-1] if optimal_options.get('calls') else {},
                        'rationale': 'Strong bullish trend with overbought RSI in high volatility',
                        'confidence': 'high'
                    })
                else:
                    strategies.append({
                        'strategy_type': 'LONG_CALL',
                        'primary_leg': atm_call,
                        'rationale': 'Strong bullish trend with overbought RSI in low volatility',
                        'confidence': 'high'
                    })
            
            # Neutral Conditions with Strong Volatility
            elif vix > self.vix_threshold:
                strategies.append({
                    'strategy_type': 'IRON_CONDOR',
                    'call_spread': {
                        'long': optimal_options.get('calls', [{}])[-1],
                        'short': atm_call
                    },
                    'put_spread': {
                        'long': optimal_options.get('puts', [{}])[-1],
                        'short': atm_put
                    },
                    'rationale': 'High volatility with neutral trend',
                    'confidence': 'medium'
                })
            
            # Default Wait Strategy
            else:
                strategies.append({
                    'strategy_type': 'WAIT',
                    'rationale': 'Market conditions unclear or not favorable',
                    'confidence': 'low'
                })

            return max(strategies, 
                      key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['confidence']])

        except Exception as e:
            logger.error(f"Error selecting primary strategy: {e}")
            return {
                'strategy_type': 'WAIT',
                'rationale': f'Error in strategy selection: {str(e)}',
                'confidence': 'low'
            }
        
    def _generate_hedge_strategy(self,
                               primary_strategy: Dict[str, Any],
                               optimal_options: Dict[str, List[Dict]],
                               is_high_volatility: bool) -> Dict[str, Any]:
        """Generate hedge recommendations based on primary strategy"""
        strategy_type = primary_strategy.get('strategy_type', '')
        
        if strategy_type in ['LONG_CALL', 'BULL_CALL_SPREAD']:
            return {
                'hedge_type': 'PROTECTIVE_PUT' if not is_high_volatility else 'PUT_SPREAD',
                'option': optimal_options.get('puts', [{}])[0],
                'sizing': '30-40% of primary position',
                'entry_timing': 'Enter hedge when delta of primary position > 0.7'
            }
        elif strategy_type in ['LONG_PUT', 'BEAR_PUT_SPREAD']:
            return {
                'hedge_type': 'COVERED_CALL' if not is_high_volatility else 'CALL_SPREAD',
                'option': optimal_options.get('calls', [{}])[0],
                'sizing': '30-40% of primary position',
                'entry_timing': 'Enter hedge when delta of primary position < -0.7'
            }
        else:
            return {
                'hedge_type': 'NONE',
                'rationale': 'Primary strategy already delta-neutral'
            }

    def _calculate_position_sizes(self,
                                optimal_options: Dict[str, List[Dict]],
                                strategy_type: str,
                                vix: float) -> Dict[str, Any]:
        """Enhanced position sizing with volatility adjustment"""
        try:
            volatility_factor = max(0.5, 1 - ((vix - self.vix_threshold) / 100))
            base_lots = 75  # Base lot size

            position_sizes = {
                'LONG_CALL': base_lots,
                'LONG_PUT': base_lots,
                'BULL_CALL_SPREAD': base_lots * 1.5,
                'BEAR_PUT_SPREAD': base_lots * 1.5,
                'IRON_CONDOR': base_lots * 0.75,
                'CALENDAR_SPREAD': base_lots,
                'WAIT': 0
            }

            lots = int(position_sizes.get(strategy_type, base_lots) * volatility_factor)

            return {
                'primary': f'{lots} lots',
                'max_positions': 1 if strategy_type in ['IRON_CONDOR', 'CALENDAR_SPREAD'] else 2,
                'scaling_rules': self._get_scaling_rules(strategy_type)
            }
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return {
                'primary': '0 lots',
                'max_positions': 0,
                'scaling_rules': 'Error in calculation'
            }
        

    def _get_scaling_rules(self, strategy_type: str) -> str:
        """Get specific scaling rules for each strategy type"""
        rules = {
            'LONG_CALL': 'Scale in 2-3 parts on dips',
            'LONG_PUT': 'Scale in 2-3 parts on rallies',
            'BULL_CALL_SPREAD': 'Enter full spread position at once',
            'BEAR_PUT_SPREAD': 'Enter full spread position at once',
            'IRON_CONDOR': 'Enter all legs simultaneously',
            'CALENDAR_SPREAD': 'Enter full position at once',
            'WAIT': 'No scaling needed'
        }
        return rules.get(strategy_type, 'Enter full position at once')

    def _generate_risk_parameters(self, vix: float) -> Dict[str, Any]:
        """Generate risk management parameters based on market conditions"""
        volatility_factor = vix / self.vix_threshold
        
        return {
            'position_loss_limit': f'{min(15 * volatility_factor, 25)}%',
            'daily_loss_limit': f'{min(5 * volatility_factor, 10)}%',
            'profit_taking': {
                'first_target': f'{20 * volatility_factor}%',
                'final_target': f'{35 * volatility_factor}%'
            },
            'stop_loss': {
                'initial': f'{10 * volatility_factor}%',
                'trailing': f'{15 * volatility_factor}%'
            }
        }

    def _generate_execution_guidelines(self,
                                    strategy_type: str,
                                    is_high_volatility: bool) -> Dict[str, Any]:
        """Generate specific execution guidelines based on strategy"""
        return {
            'entry_conditions': self._get_entry_conditions(strategy_type),
            'exit_conditions': self._get_exit_conditions(strategy_type),
            'trade_management': self._get_trade_management(strategy_type, is_high_volatility)
        }

    def _get_entry_conditions(self, strategy_type: str) -> List[str]:
        """Get specific entry conditions for each strategy type"""
        conditions = {
            'LONG_CALL': [
                'Price near support levels',
                'RSI < 60',
                'Positive underlying momentum'
            ],
            'LONG_PUT': [
                'Price near resistance levels',
                'RSI > 40',
                'Negative underlying momentum'
            ],
            'BULL_CALL_SPREAD': [
                'Price above short-term moving average',
                'Implied volatility relatively high',
                'Positive sector momentum'
            ],
            'BEAR_PUT_SPREAD': [
                'Price below short-term moving average',
                'Implied volatility relatively high',
                'Negative sector momentum'
            ]
        }
        
        return conditions.get(strategy_type, [
            'Default entry conditions',
            'Check all legs for liquidity',
            'Monitor implied volatility skew'
        ])

    def _get_exit_conditions(self, strategy_type: str) -> List[str]:
        """Get specific exit conditions for each strategy type"""
        return [
            'Target profit reached',
            'Stop loss hit',
            'Technical trend reversal',
            'Significant volatility change'
        ]

    def _get_trade_management(self,
                            strategy_type: str,
                            is_high_volatility: bool) -> Dict[str, Any]:
        """Get trade management guidelines based on strategy and volatility"""
        return {
            'position_review_frequency': 'Hourly' if is_high_volatility else 'Daily',
            'adjustment_triggers': [
                'Delta neutrality breach > 20%',
                'Implied volatility significant change',
                'Technical trend reversal'
            ]
        }





# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
public_url = None



def validate_payload(market_data: Dict[str, Any]) -> bool:
    """
    Validate the incoming market data payload

    Args:
        market_data (Dict): Market data payload to validate

    Returns:
        bool: True if payload is valid, False otherwise
    """
    try:
        required_keys = ['historical_data', 'current_market']
        if not all(key in market_data for key in required_keys):
            logger.warning("Missing required keys")
            return False

        index_data = market_data.get('historical_data', {}).get('index', [])
        if not index_data or not all(key in index_data[0].get('price_data', {})
                                   for key in ['open', 'high', 'low', 'close']):
            logger.warning("Invalid price data structure")
            return False

        return True
    except Exception as e:
        logger.error(f"Payload validation error: {e}")
        return False

@app.route('/')
def home():
    """
    Home route to check application status
    
    Returns:
        JSON: Application status
    """
    return jsonify({
        'status': 'success',
        'message': 'Options Analysis API is running'
    })

@app.route('/analyse', methods=['POST'])
def analyze_data():
    """
    Main route for market data analysis

    Returns:
        JSON: Comprehensive market analysis results
    """
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({'status': 'error', 'message': 'No payload received'}), 400

        analysis_results = perform_market_analysis(payload)
        return jsonify({'status': 'success', 'data': analysis_results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

class MarketAnalyzer:
    """
    Primary class for conducting market structure analysis
    """
    def __init__(self, market_data: Dict[str, Any]):
        """
        Initialize MarketAnalyzer with market data

        Args:
            market_data (Dict): Comprehensive market data dictionary
        """
        self.market_data = market_data
        self.current_market = market_data.get('current_market', {})
        self.historical_data = market_data.get('historical_data', {})
        self.options_data = market_data.get('options', {})

    def analyze_market_structure(self) -> Dict[str, Any]:
        """
        Analyze overall market structure and key metrics

        Returns:
            Dict: Comprehensive market structure analysis
        """
        try:
            current_market = self.current_market.get('index', {})
            vix_data = self.current_market.get('vix', {})

            return {
                'price_levels': {
                    'current': current_market.get('ltp', 0),
                    'high': current_market.get('high', 0),
                    'low': current_market.get('low', 0),
                    'open': current_market.get('open', 0),
                    'prev_close': current_market.get('close', 0)
                },
                'trend_analysis': {
                    'intraday': {
                        'change': current_market.get('ltp', 0) - current_market.get('open', 0),
                        'change_percent': current_market.get('percentChange', 0),
                        'direction': 'Bullish' if current_market.get('ltp', 0) >= current_market.get('open', 0) else 'Bearish'
                    },
                    'overall': {
                        'net_change': current_market.get('netChange', 0),
                        'net_change_percent': current_market.get('percentChange', 0)
                    }
                },
                'volatility': {
                    'market_range': {
                        'day_high': current_market.get('high', 0),
                        'day_low': current_market.get('low', 0),
                        'range_percent': abs(current_market.get('high', 0) - current_market.get('low', 0)) /
                                       current_market.get('low', 1) * 100
                    },
                    'vix_current': vix_data.get('ltp', 0),
                    'vix_change': vix_data.get('netChange', 0),
                    'vix_percent_change': vix_data.get('percentChange', 0)
                }
            }
        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            return {}

def analyze_technical_indicators(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform technical analysis using pandas and custom indicators

    Args:
        data (Dict): Market data dictionary

    Returns:
        Dict: Technical analysis results with converted data types
    """
    try:
        index_history = data.get('historical_data', {}).get('index', [])

        if not index_history:
            return {'error': 'Insufficient historical data'}

        # Create DataFrame with explicit type conversion
        df = pd.DataFrame([{
            'timestamp': pd.to_datetime(entry['timestamp']),
            'open': float(entry['price_data']['open']),  # Explicit float conversion
            'high': float(entry['price_data']['high']),
            'low': float(entry['price_data']['low']),
            'close': float(entry['price_data']['close'])
        } for entry in index_history])

        df = df.sort_values('timestamp')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].interpolate(method='linear')

        # Calculate moving averages with explicit conversion
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()

        # Calculate momentum indicators
        momentum = df['close'].pct_change(periods=10)
        rsi = calculate_rsi(df['close'])

        # Perform candlestick pattern analysis
        candlestick_patterns = analyze_candlestick_patterns(df)

        # Safely extract last values with type conversion and handling of NaN
        last_close = float(df['close'].iloc[-1]) if not pd.isna(df['close'].iloc[-1]) else 0.0
        last_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0.0
        last_sma_20 = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else last_close
        last_sma_50 = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else last_close
        last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # Trend analysis with explicit type conversion
        trend_strength = 'Strong' if abs(last_momentum) > 0.05 else 'Moderate'
        trend_direction = 'Bullish' if last_momentum > 0 else 'Bearish'

        return {
            'moving_averages': {
                'sma_20': float(last_sma_20),  # Explicit float conversion
                'sma_50': float(last_sma_50),
                'price_vs_sma20': float(last_close - last_sma_20),
                'price_vs_sma50': float(last_close - last_sma_50)
            },
            'momentum_indicators': {
                '10_day_momentum': float(last_momentum),
                'rsi': float(last_rsi),
                'trend_strength': trend_strength,
                'trend_direction': trend_direction
            },
            'candlestick_analysis': {
                # Convert any numpy values to standard types
                k: int(v) if isinstance(v, (np.integer, np.int64)) else v
                for k, v in candlestick_patterns.items()
            },
            'additional_insights': {
                'total_data_points': int(len(df)),
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                }
            }
        }
    except Exception as e:
        return {
            'error': 'Technical analysis processing error',
            'details': str(e)
        }

def calculate_rsi(price_series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series

    Args:
        price_series (pd.Series): Price series data
        window (int): Rolling window size for RSI calculation

    Returns:
        pd.Series: RSI values
    """
    delta = price_series.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_candlestick_patterns(price_data):
    """
    Analyze candlestick patterns in the given price data

    Args:
        price_data (pd.DataFrame): Price data with columns 'open', 'high', 'low', 'close'

    Returns:
        dict: Detected candlestick patterns
    """
    patterns = {}

    # Detect bullish engulfing pattern
    bullish_engulfing = (price_data['close'] > price_data['open'].shift(1)) & (price_data['open'] < price_data['close'].shift(1))
    patterns['bullish_engulfing'] = bullish_engulfing.sum()

    # Detect bearish engulfing pattern
    bearish_engulfing = (price_data['close'] < price_data['open'].shift(1)) & (price_data['open'] > price_data['close'].shift(1))
    patterns['bearish_engulfing'] = bearish_engulfing.sum()

    # Add more candlestick patterns as needed

    return patterns

def analyze_options_chain(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced options chain analysis including Greeks and index-specific handling"""
    try:
        # Initialize calculators
        greeks_calculator = OptionsGreeksCalculator()
        index_analyzer = IndexOptionsAnalyzer(greeks_calculator)
        
        # Extract required data
        current_price = market_data.get('current_market', {}).get('index', {}).get('ltp', 0)
        vix = market_data.get('current_market', {}).get('vix', {}).get('ltp', 0)
        
        # Get futures data for volume reference
        futures_data = market_data.get('futures', {}).get('near_month', {})
        
        # Get options chain data
        options_chain = {
            'calls': market_data.get('options', {}).get('calls', []),
            'puts': market_data.get('options', {}).get('puts', [])
        }
        
        # Select optimal options with Greeks
        optimal_options = index_analyzer.select_optimal_options(
            current_price,
            options_chain,
            futures_data,
            vix
        )
        
        # Calculate aggregate metrics
        call_oi = sum(opt.get('openInterest', 0) for opt in options_chain['calls'])
        put_oi = sum(opt.get('openInterest', 0) for opt in options_chain['puts'])
        
        # Use futures volume for index
        total_volume = futures_data.get('volume', 0)
        
        return {
            'activity_analysis': {
                'call_oi': call_oi,
                'put_oi': put_oi,
                'futures_volume': total_volume
            },
            'optimal_options': {
                'calls': [{
                    'strike': opt['strike_price'],
                    'ltp': opt.get('ltp', 0),
                    'greeks': opt['greeks'],
                    'entry_zones': opt['entry_zones'],
                    'liquidity_score': opt['liquidity_score']
                } for opt in optimal_options['calls']],
                'puts': [{
                    'strike': opt['strike_price'],
                    'ltp': opt.get('ltp', 0),
                    'greeks': opt['greeks'],
                    'entry_zones': opt['entry_zones'],
                    'liquidity_score': opt['liquidity_score']
                } for opt in optimal_options['puts']]
            },
            'put_call_ratios': {
                'oi_pcr': put_oi / call_oi if call_oi > 0 else 0,
                'volume_pcr': futures_data.get('put_volume', 0) / futures_data.get('call_volume', 1)
            },
            'market_metrics': {
                'total_oi': {
                    'calls': call_oi,
                    'puts': put_oi
                },
                'total_volumes': {
                    'calls': futures_data.get('call_volume', 0),
                    'puts': futures_data.get('put_volume', 0)
                },
                'oi_pcr': put_oi / call_oi if call_oi > 0 else 0,
                'volume_pcr': futures_data.get('put_volume', 0) / futures_data.get('call_volume', 1)
            },
            'market_warning': _check_market_warnings(put_oi, call_oi, vix)
        }
    except Exception as e:
        logger.error(f"Options chain analysis error: {e}")
        return {}

def _check_market_warnings(put_oi: float, call_oi: float, vix: float) -> Dict[str, Any]:
    """Generate market warnings based on extreme values"""
    warnings = {}
    
    # Check PCR ratio
    if call_oi > 0:
        pcr = put_oi / call_oi
        if pcr > 10:
            warnings['extreme_put_activity'] = {
                'type': 'High Put/Call Ratio',
                'value': pcr,
                'threshold': 10,
                'implication': 'Extreme bearish sentiment, potential reversal point'
            }
        elif pcr < 0.1:
            warnings['extreme_call_activity'] = {
                'type': 'Low Put/Call Ratio',
                'value': pcr,
                'threshold': 0.1,
                'implication': 'Extreme bullish sentiment, potential reversal point'
            }
    
    # Check VIX
    if vix > 30:
        warnings['high_volatility'] = {
            'type': 'Elevated VIX',
            'value': vix,
            'threshold': 30,
            'implication': 'High market uncertainty, consider reduced position sizes'
        }
    
    return warnings


def generate_market_summary(market_structure: Dict[str, Any],
                          options_analysis: Dict[str, Any],
                          technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive market summary

    Args:
        market_structure (Dict): Market structure analysis
        options_analysis (Dict): Options market analysis
        technical_analysis (Dict): Technical analysis results

    Returns:
        Dict: Market summary and sentiment
    """
    try:
        intraday_trend = market_structure.get('trend_analysis', {}).get('intraday', {}).get('direction', 'Neutral')
        oi_pcr = options_analysis.get('put_call_ratios', {}).get('oi_pcr', 1)
        options_sentiment = 'Bearish' if oi_pcr < 0.7 else 'Bullish' if oi_pcr > 1.3 else 'Neutral'
        vix_value = market_structure.get('volatility', {}).get('vix_current', 0)
        volatility_state = 'High' if vix_value > 20 else 'Low' if vix_value < 12 else 'Normal'

        candlestick_sentiment = 'Bullish' if technical_analysis.get('candlestick_analysis', {}).get('bullish_engulfing', 0) > 0 else 'Bearish'

        return {
            'market_bias': intraday_trend,
            'options_sentiment': options_sentiment,
            'technical_bias': intraday_trend,
            'volatility_state': volatility_state,
            'candlestick_sentiment': candlestick_sentiment
        }
    except Exception as e:
        logger.error(f"Market summary generation error: {e}")
        return {}

def _analyze_price_action(market_structure: Dict) -> Dict:
    """
    Analyze price action and market movement characteristics

    Args:
        market_structure (Dict): Market structure analysis data

    Returns:
        Dict: Price action analysis insights
    """
    try:
        price_levels = market_structure.get('price_levels', {})
        trend_analysis = market_structure.get('trend_analysis', {}).get('intraday', {})
        volatility = market_structure.get('volatility', {})

        return {
            'price_movement': trend_analysis.get('direction', 'Neutral'),
            'change_percent': trend_analysis.get('change_percent', 0),
            'daily_range_percent': volatility.get('market_range', {}).get('range_percent', 0),
            'intraday_volatility': 'High' if abs(trend_analysis.get('change_percent', 0)) > 2 else 'Normal'
        }
    except Exception as e:
        logger.error(f"Price action analysis error: {e}")
        return {}

def _generate_primary_signal(signal_strength: Dict) -> str:
    """
    Generate a primary trading signal based on multiple analysis perspectives

    Args:
        signal_strength (Dict): Strength indicators from various analysis perspectives

    Returns:
        str: Primary trading recommendation
    """
    # Ensure the function body is not empty
    technical_signal = signal_strength.get('technical', {}).get('momentum_signal', 'Neutral')
    options_pcr = signal_strength.get('options', {}).get('put_call_ratio', 1)
    price_action = signal_strength.get('price_action', {}).get('price_movement', 'Neutral')

    # Complex signal generation logic
    signals = [technical_signal, price_action]

    # Count occurrences
    buy_count = signals.count('Buy')
    sell_count = signals.count('Sell')

    # Additional PCR-based adjustment
    if options_pcr < 0.7:  # Extremely bearish options sentiment
        sell_count += 1
    elif options_pcr > 1.3:  # Extremely bullish options sentiment
        buy_count += 1

    # Determine primary signal
    if buy_count > sell_count:
        return 'Strong Buy'
    elif sell_count > buy_count:
        return 'Strong Sell'
    else:
        return 'Neutral/Hold'

def generate_trading_signals(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive trading signals incorporating Greeks and index-specific analysis"""
    try:
        # Initialize strategy generator
        strategy_generator = OptionsStrategyGenerator()
        
        # Extract required components from analysis
        market_structure = analysis_result.get('market_structure', {})
        technical = analysis_result.get('technical_analysis', {})
        options = analysis_result.get('options_analysis', {})
        
        # Generate options-specific trading strategy
        options_strategy = strategy_generator.generate_trading_strategy(
            market_data=analysis_result,
            optimal_options=options.get('optimal_options', {}),
            technical_analysis=technical
        )
        
        # Calculate trade levels incorporating Greeks
        trade_levels = _calculate_advanced_trade_levels(
            market_structure,
            technical,
            options
        )
        
        return {
            'options_strategy': options_strategy,
            'trade_levels': trade_levels,
            'risk_summary': _generate_risk_summary(options_strategy, market_structure),
            'execution_plan': _create_execution_plan(options_strategy, trade_levels)
        }
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return {}

def _calculate_advanced_trade_levels(
    market_structure: Dict[str, Any],
    technical: Dict[str, Any],
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate advanced trade levels incorporating Greeks and option chain structure"""
    current_price = market_structure.get('price_levels', {}).get('current', 0)
    vix = market_structure.get('volatility', {}).get('vix_current', 0)
    
    # Get optimal options
    optimal_options = options.get('optimal_options', {})
    
    # Calculate support and resistance incorporating option chain analysis
    support_resistance = _calculate_options_based_levels(
        current_price,
        optimal_options,
        vix
    )
    
    # Define entry and exit zones based on Greeks
    entry_zones = {}
    for option_type in ['calls', 'puts']:
        if optimal_options.get(option_type):
            primary_option = optimal_options[option_type][0]
            entry_zones[option_type] = {
                'primary': primary_option.get('entry_zones', {}),
                'greeks_threshold': {
                    'delta': 0.4 if option_type == 'calls' else -0.4,
                    'theta': -0.1,
                    'gamma': 0.02
                }
            }
    
    return {
        'price_levels': {
            'current': current_price,
            'support_resistance': support_resistance
        },
        'entry_zones': entry_zones,
        'exit_criteria': _generate_exit_criteria(vix)
    }

def _calculate_options_based_levels(
    current_price: float,
    optimal_options: Dict[str, List[Dict]],
    vix: float
) -> Dict[str, Any]:
    """Calculate support and resistance levels using options chain data"""
    volatility_factor = vix / 20  # Normalize around VIX of 20
    
    # Extract strike prices
    call_strikes = [opt.get('strike', 0) for opt in optimal_options.get('calls', [])]
    put_strikes = [opt.get('strike', 0) for opt in optimal_options.get('puts', [])]
    
    # Find nearest strikes
    nearest_call = min(call_strikes, key=lambda x: abs(x - current_price)) if call_strikes else current_price
    nearest_put = min(put_strikes, key=lambda x: abs(x - current_price)) if put_strikes else current_price
    
    return {
        'strong_support': min(put_strikes) if put_strikes else (current_price * 0.95),
        'weak_support': nearest_put * (1 - 0.01 * volatility_factor),
        'weak_resistance': nearest_call * (1 + 0.01 * volatility_factor),
        'strong_resistance': max(call_strikes) if call_strikes else (current_price * 1.05),
        'expected_range': {
            'lower': current_price * (1 - 0.02 * volatility_factor),
            'upper': current_price * (1 + 0.02 * volatility_factor)
        }
    }

def _generate_exit_criteria(vix: float) -> Dict[str, Any]:
    """Generate dynamic exit criteria based on market conditions"""
    volatility_factor = vix / 20
    
    return {
        'stop_loss': {
            'percentage': 15 * volatility_factor,
            'greeks_based': {
                'delta_threshold': 0.8,
                'gamma_threshold': 0.04
            }
        },
        'profit_taking': {
            'first_target': 20 * volatility_factor,
            'final_target': 35 * volatility_factor
        },
        'time_based': {
            'minimum_hold': '2 hours',
            'maximum_hold': '2 days' if vix > 20 else '5 days'
        }
    }

def _generate_risk_summary(options_strategy: Dict[str, Any], market_structure: Dict[str, Any]) -> Dict[str, Any]:
    """Generate risk summary for the trading strategy"""
    vix = market_structure.get('volatility', {}).get('vix_current', 0)
    
    return {
        'market_risk': {
            'volatility_regime': 'High' if vix > 20 else 'Normal' if vix > 15 else 'Low',
            'risk_rating': _calculate_risk_rating(options_strategy, vix)
        },
        'position_risk': {
            'max_loss': options_strategy.get('risk_parameters', {}).get('position_loss_limit', 'N/A'),
            'margin_requirements': _calculate_margin_requirements(options_strategy),
            'greeks_exposure': _summarize_greeks_exposure(options_strategy)
        }
    }

def _calculate_risk_rating(strategy: Dict[str, Any], vix: float) -> str:
    """Calculate risk rating based on strategy and market conditions"""
    strategy_type = strategy.get('primary_strategy', {}).get('strategy_type', '')
    
    strategy_risk = {
        'LONG_CALL': 3,
        'LONG_PUT': 3,
        'BULL_CALL_SPREAD': 2,
        'BEAR_PUT_SPREAD': 2,
        'IRON_CONDOR': 1,
        'CALENDAR_SPREAD': 2
    }
    
    base_risk = strategy_risk.get(strategy_type, 2)
    volatility_factor = vix / 20
    
    total_risk = base_risk * volatility_factor
    
    if total_risk > 2.5:
        return 'High'
    elif total_risk > 1.5:
        return 'Medium'
    else:
        return 'Low'

def _calculate_margin_requirements(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate margin requirements for the options strategy"""
    strategy_type = strategy.get('primary_strategy', {}).get('strategy_type', '')
    
    margin_multipliers = {
        'LONG_CALL': 1.0,  # Full premium
        'LONG_PUT': 1.0,   # Full premium
        'BULL_CALL_SPREAD': 0.5,  # Difference between strikes
        'BEAR_PUT_SPREAD': 0.5,
        'IRON_CONDOR': 0.3,
        'CALENDAR_SPREAD': 0.7
    }
    
    multiplier = margin_multipliers.get(strategy_type, 1.0)
    
    return {
        'initial_margin': f"{multiplier * 100}% of position value",
        'maintenance_margin': f"{multiplier * 75}% of position value",
        'margin_calls': "Based on daily MTM settlement"
    }

def _summarize_greeks_exposure(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize Greeks exposure for the overall position"""
    primary_leg = strategy.get('primary_strategy', {}).get('primary_leg', {})
    greeks = primary_leg.get('greeks', {})
    
    return {
        'delta_exposure': _categorize_greek_exposure(greeks.get('delta', 0)),
        'gamma_risk': _categorize_greek_exposure(greeks.get('gamma', 0)),
        'theta_decay': _categorize_greek_exposure(greeks.get('theta', 0)),
        'vega_risk': _categorize_greek_exposure(greeks.get('vega', 0))
    }

def _categorize_greek_exposure(value: float) -> str:
    """Categorize Greeks exposure levels"""
    abs_value = abs(value)
    
    if abs_value > 0.7:
        return 'High'
    elif abs_value > 0.3:
        return 'Medium'
    else:
        return 'Low'

def _create_execution_plan(strategy: Dict[str, Any], trade_levels: Dict[str, Any]) -> Dict[str, Any]:
    """Create detailed execution plan for the strategy"""
    return {
        'entry_plan': {
            'primary_entry': strategy.get('primary_strategy', {}),
            'entry_zones': trade_levels.get('entry_zones', {}),
            'execution_steps': strategy.get('execution_guidelines', {})
        },
        'management_plan': {
            'monitoring_frequency': 'Hourly',
            'adjustment_triggers': _get_adjustment_triggers(strategy),
            'roll_conditions': _get_roll_conditions(strategy)
        },
        'exit_plan': {
            'profit_targets': trade_levels.get('exit_criteria', {}).get('profit_taking', {}),
            'stop_losses': trade_levels.get('exit_criteria', {}).get('stop_loss', {}),
            'time_stops': trade_levels.get('exit_criteria', {}).get('time_based', {})
        }
    }

def _get_adjustment_triggers(strategy: Dict[str, Any]) -> List[str]:
    """Define specific triggers for position adjustments"""
    return [
        "Delta neutrality breach > 20%",
        "Implied volatility change > 20%",
        "Technical trend reversal",
        "Time decay acceleration",
        "Hedge position delta change"
    ]

def _get_roll_conditions(strategy: Dict[str, Any]) -> List[str]:
    """Define conditions for rolling options positions"""
    return [
        "Less than 5 days to expiry",
        "Delta moves beyond acceptable range",
        "Significant implied volatility change",
        "Better opportunities in different strikes"
    ]

def _analyze_options_strength(options: Dict) -> Dict:
    """
    Analyze options market strength and sentiment

    Args:
        options (Dict): Options market analysis

    Returns:
        Dict: Options market strength assessment
    """
    try:
        call_oi = options['activity_analysis']['call_oi']
        put_oi = options['activity_analysis']['put_oi']
        oi_pcr = options['put_call_ratios']['oi_pcr']

        return {
            'put_call_ratio': oi_pcr,
            'open_interest_balance': {
                'calls': call_oi,
                'puts': put_oi,
                'net_sentiment': 'Bullish' if call_oi > put_oi else 'Bearish'
            }
        }
    except Exception as e:
        logger.error(f"Options strength analysis error: {e}")
        return {}

def _suggest_option_strategies(options: Dict, market: Dict) -> List[Dict]:
    """
    Suggest option trading strategies based on market conditions

    Args:
        options (Dict): Options market analysis data
        market (Dict): Market structure data

    Returns:
        List[Dict]: Recommended option strategies
    """
    vix = market['volatility']['vix_current']
    pcr = options['put_call_ratios']['oi_pcr']
    current_price = market['price_levels']['current']

    strategies = []

    # High Volatility Strategies
    if vix > 20:
        strategies.extend([
            {
                'strategy': 'Iron Condor',
                'rationale': 'High volatility environment suggests range-bound trading',
                'max_risk': f'±{vix * 0.5}%',
                'recommendation_strength': 'Strong'
            },
            {
                'strategy': 'Short Straddle',
                'rationale': 'Implied volatility suggests potential price stabilization',
                'max_risk': f'±{vix * 0.75}%',
                'recommendation_strength': 'Medium'
            }
        ])

    # Put Call Ratio Based Strategies
    if pcr > 1.5:
        strategies.append({
            'strategy': 'Bull Put Spread',
            'rationale': 'Extreme bearish sentiment suggests potential market reversal',
            'entry_price_range': f'{current_price * 0.95} - {current_price * 1.05}',
            'recommendation_strength': 'Medium'
        })
    elif pcr < 0.5:
        strategies.append({
            'strategy': 'Bear Call Spread',
            'rationale': 'Extreme bullish sentiment suggests potential market correction',
            'entry_price_range': f'{current_price * 0.95} - {current_price * 1.05}',
            'recommendation_strength': 'Medium'
        })

    # Low Volatility Conservative Strategies
    if vix < 12:
        strategies.append({
            'strategy': 'Covered Call',
            'rationale': 'Low volatility environment suggests conservative income generation',
            'potential_income': f'{0.5 * vix}%',
            'recommendation_strength': 'Low Risk'
        })

    return strategies


def _calculate_trade_levels(market_structure: Dict, technical_analysis: Dict) -> Dict:
    """
    Calculate key trading levels and support/resistance zones
    Args:
        market_structure (Dict): Market structure data
        technical_analysis (Dict): Technical analysis data

    Returns:
        Dict: Trading levels and potential entry/exit points
    """
    try:
        current_price = market_structure['price_levels']['current']
        sma_20 = technical_analysis['moving_averages']['sma_20']
        sma_50 = technical_analysis['moving_averages']['sma_50']
        rsi = technical_analysis['momentum_indicators']['rsi']

        # Calculate support and resistance levels based on moving averages
        support_levels = {
            'primary': sma_20,
            'secondary': sma_50
        }
        resistance_levels = {
            'day_high': market_structure['price_levels']['high'],
            'immediate_resistance': sma_20 * 1.02  # 2% above SMA
        }

        # Determine entry zones based on RSI and moving averages
        if rsi < 30:  # Oversold condition
            entry_zones = {
                'bullish_entry': sma_20 * 0.98,  # 2% below SMA
            }
        elif rsi > 70:  # Overbought condition
            entry_zones = {
                'bearish_entry': sma_20 * 1.02  # 2% above SMA
            }
        else:
            entry_zones = {
                'bullish_entry': sma_20 * 0.98,
                'bearish_entry': sma_20 * 1.02
            }

        return {
            'current_price': current_price,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'entry_zones': entry_zones
        }
    except Exception as e:
        logger.error(f"Trade levels calculation error: {e}")
        return {}

def _analyze_technical_strength(technical: Dict) -> Dict:
    """
    Analyze technical indicators strength

    Args:
        technical (Dict): Technical analysis results

    Returns:
        Dict: Technical strength assessment
    """
    sma_20 = technical['moving_averages']['sma_20']
    sma_50 = technical['moving_averages']['sma_50']
    momentum = technical['momentum_indicators']['10_day_momentum']

    return {
        'trend_strength': 'Strong' if abs(momentum) > 0.01 else 'Weak',
        'ma_alignment': 'Bullish' if sma_20 > sma_50 else 'Bearish',
        'momentum_signal': 'Buy' if momentum > 0 else 'Sell'
    }
class MarketDataValidator:
    """
    Comprehensive data validation and preprocessing for market analysis.
    This class ensures data quality and enhances market signals for more reliable analysis.
    """
    @staticmethod
    def validate_options_data(options_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean options market data, with special handling for index options.
        
        This method performs several key validations:
        1. Checks and corrects for zero active options
        2. Validates put/call ratios
        3. Handles missing or anomalous data
        4. Adds market warnings for extreme conditions
        
        Args:
            options_data: Dictionary containing options market data
            
        Returns:
            Enhanced and validated options data dictionary
        """
        try:
            # Check and correct zero active options
            if (options_data.get('option_chain_structure', {}).get('active_calls', 0) == 0 or
                options_data.get('option_chain_structure', {}).get('active_puts', 0) == 0):

                # Reconstruct active options from available data
                calls = options_data.get('calls', [])
                puts = options_data.get('puts', [])

                # Count truly active options by checking open interest
                active_calls = len([c for c in calls if c.get('openInterest', 0) > 0])
                active_puts = len([p for p in puts if p.get('openInterest', 0) > 0])

                # Update the option chain structure with accurate counts
                options_data['option_chain_structure'] = {
                    'active_calls': active_calls,
                    'active_puts': active_puts,
                    'total_strikes': len(set(
                        [c.get('strikePrice', 0) for c in calls] +
                        [p.get('strikePrice', 0) for p in puts]
                    ))
                }

            # Calculate and validate Put-Call Ratios
            oi_pcr = options_data.get('put_call_ratios', {}).get('oi_pcr', 0)
            volume_pcr = options_data.get('put_call_ratios', {}).get('volume_pcr', 0)

            # Flag extreme ratios that might indicate unusual market conditions
            if oi_pcr > 10 or volume_pcr > 100:
                options_data['market_warning'] = {
                    'type': 'Extreme Put Activity',
                    'oi_pcr': oi_pcr,
                    'volume_pcr': volume_pcr,
                    'recommendation': 'Investigate underlying market sentiment'
                }

            return options_data

        except Exception as e:
            logger.error(f"Options data validation error: {e}")
            return {}

    @staticmethod
    def enhance_market_signals(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance and refine market signals with additional context and analysis.
        
        This method adds:
        1. Confidence levels to signals
        2. Detailed trend analysis
        3. Multi-timeframe perspectives
        4. Market sentiment indicators
        
        Args:
            market_data: Dictionary containing market analysis data
            
        Returns:
            Enhanced market data with additional signals and context
        """
        try:
            # Extract technical analysis components
            technical = market_data.get('technical_analysis', {})
            momentum = technical.get('momentum_indicators', {})

            # Calculate Signal Confidence Components
            signal_confidence = {
                'momentum_strength': abs(momentum.get('10_day_momentum', 0)),
                'trend_consistency': 1 if momentum.get('trend_direction') == 'Bearish' else 0,
                'moving_average_divergence': abs(
                    technical.get('moving_averages', {}).get('price_vs_sma50', 0) /
                    technical.get('moving_averages', {}).get('sma_50', 1)
                ) * 100
            }

            # Calculate Comprehensive Signal Score
            signal_score = (
                signal_confidence['momentum_strength'] * 0.4 +
                signal_confidence['trend_consistency'] * 0.3 +
                signal_confidence['moving_average_divergence'] * 0.3
            )

            # Generate Enhanced Market Signals
            market_data['enhanced_signals'] = {
                'signal_confidence': signal_score,
                'confidence_level': (
                    'High' if signal_score > 0.7 else
                    'Medium' if signal_score > 0.4 else
                    'Low'
                ),
                'recommended_position': (
                    'Short' if signal_score > 0.7 else
                    'Neutral' if signal_score > 0.4 else
                    'Wait'
                ),
                'market_sentiment': _calculate_market_sentiment(market_data),
                'volume_analysis': _analyze_volume_patterns(market_data)
            }

            return market_data

        except Exception as e:
            logger.error(f"Market signal enhancement error: {e}")
            return market_data

    @staticmethod
    def validate_futures_data(futures_data: Dict[str, Any], 
                            index_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance futures data, particularly important for index trading.
        
        This method:
        1. Verifies futures data consistency
        2. Calculates proper volume metrics
        3. Adds basis analysis
        4. Handles rollovers
        
        Args:
            futures_data: Dictionary containing futures market data
            index_data: Dictionary containing index market data
            
        Returns:
            Validated and enhanced futures data
        """
        try:
            if not futures_data:
                return {}

            # Calculate basis (difference between futures and spot)
            spot_price = index_data.get('ltp', 0)
            futures_price = futures_data.get('ltp', 0)
            basis = futures_price - spot_price
            basis_percentage = (basis / spot_price * 100) if spot_price else 0

            # Enhance futures data with calculated metrics
            enhanced_futures = {
                **futures_data,
                'basis_analysis': {
                    'absolute_basis': basis,
                    'basis_percentage': basis_percentage,
                    'basis_status': 'Premium' if basis > 0 else 'Discount',
                    'basis_trend': _analyze_basis_trend(basis_percentage)
                },
                'volume_metrics': {
                    'normalized_volume': futures_data.get('volume', 0) / 100000,  # Convert to lakhs
                    'volume_trend': _analyze_volume_trend(futures_data)
                },
                'rollover_metrics': _calculate_rollover_metrics(futures_data)
            }

            return enhanced_futures

        except Exception as e:
            logger.error(f"Futures data validation error: {e}")
            return {}

def _calculate_market_sentiment(market_data: Dict[str, Any]) -> Dict[str, str]:
    """Calculate overall market sentiment using multiple indicators"""
    technical = market_data.get('technical_analysis', {})
    momentum = technical.get('momentum_indicators', {})
    rsi = momentum.get('rsi', 50)
    trend = momentum.get('trend_direction', 'Neutral')
    
    sentiment = {
        'primary': 'Bullish' if rsi > 60 and trend == 'Bullish' else
                  'Bearish' if rsi < 40 and trend == 'Bearish' else
                  'Neutral',
        'strength': 'Strong' if abs(rsi - 50) > 20 else
                   'Moderate' if abs(rsi - 50) > 10 else
                   'Weak',
        'sustainability': 'High' if 30 < rsi < 70 else 'Low'
    }
    
    return sentiment

def _analyze_volume_patterns(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze volume patterns for market strength confirmation"""
    volume = market_data.get('volume', 0)
    avg_volume = market_data.get('average_volume', 0)
    
    return {
        'volume_strength': 'High' if volume > avg_volume * 1.5 else
                          'Low' if volume < avg_volume * 0.5 else
                          'Normal',
        'trend_confirmation': 'Confirmed' if volume > avg_volume else 'Weak'
    }

def _analyze_basis_trend(basis_percentage: float) -> str:
    """Analyze the trend in futures basis"""
    if basis_percentage > 0.5:
        return 'Strong Premium'
    elif basis_percentage > 0:
        return 'Mild Premium'
    elif basis_percentage < -0.5:
        return 'Strong Discount'
    else:
        return 'Mild Discount'

def _analyze_volume_trend(futures_data: Dict[str, Any]) -> str:
    """Analyze the trend in futures volume"""
    volume = futures_data.get('volume', 0)
    avg_volume = futures_data.get('average_volume', 1)
    
    volume_ratio = volume / avg_volume if avg_volume else 0
    
    if volume_ratio > 1.5:
        return 'Strongly Rising'
    elif volume_ratio > 1.2:
        return 'Rising'
    elif volume_ratio < 0.8:
        return 'Falling'
    else:
        return 'Stable'

def _calculate_rollover_metrics(futures_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate rollover metrics for futures contracts"""
    return {
        'rollover_percentage': futures_data.get('rollover_percentage', 0),
        'rollover_cost': futures_data.get('rollover_cost', 0),
        'days_to_expiry': futures_data.get('days_to_expiry', 0)
    }

def prepare_advanced_trading_strategy(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Develop a sophisticated, adaptive trading strategy

    Key Components:
    - Dynamic risk management
    - Adaptive position sizing
    - Multi-condition entry/exit criteria
    - Game theory principles
    """
    try:
        # Validate and enhance market data
        validated_data = MarketDataValidator.validate_options_data(market_data.get('options_analysis', {}))
        enhanced_data = MarketDataValidator.enhance_market_signals(market_data)

        # Extract key parameters
        options_analysis = validated_data
        market_structure = market_data.get('market_structure', {})
        enhanced_signals = enhanced_data.get('enhanced_signals', {})

        # Risk Management Parameters
        current_price = market_structure.get('price_levels', {}).get('current', 0)
        volatility = market_structure.get('volatility', {}).get('vix_current', 0)

        # Adaptive Position Sizing
        base_risk_unit = current_price * 0.01  # 1% of current price
        volatility_adjustment = 1 + (volatility / 100)
        position_size = base_risk_unit * volatility_adjustment

        # Game Theory Principles
        market_sentiment = enhanced_signals.get('market_sentiment', 'Neutral')
        participant_behavior = analyze_participant_behavior(market_data)
        optimal_strategy = determine_optimal_strategy(market_sentiment, participant_behavior)

        # Advanced Trading Strategy
        trading_strategy = {
            'risk_management': {
                'max_position_size': position_size,
                'stop_loss_percentage': 2 * (volatility / 100),
                'take_profit_percentage': 3 * (volatility / 100)
            },
            'entry_criteria': {
                'primary_condition': enhanced_signals.get('recommended_position', 'Wait'),
                'confidence_level': enhanced_signals.get('confidence_level', 'Low'),
                'volatility_threshold': volatility > 15,  # VIX condition
                'price_action_confirmation': abs(
                    market_structure.get('trend_analysis', {})
                    .get('intraday', {})
                    .get('change_percent', 0)
                ) > 0.5
            },
            'game_theory_strategy': optimal_strategy,
            'recommended_actions': {
                'options_strategy': (
                    'Bull Put Spread' if enhanced_signals.get('recommended_position') == 'Neutral'
                    else 'Short Put' if enhanced_signals.get('recommended_position') == 'Short'
                    else 'No Action'
                ),
                'position_direction': (
                    'Neutral' if enhanced_signals.get('recommended_position') == 'Neutral'
                    else 'Bearish' if enhanced_signals.get('recommended_position') == 'Short'
                    else 'Wait'
                )
            }
        }

        return {
            'validated_market_data': validated_data,
            'enhanced_signals': enhanced_signals,
            'trading_strategy': trading_strategy
        }

    except Exception as e:
        logger.error(f"Advanced trading strategy development error: {e}")
        return {}

def analyze_participant_behavior(market_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze market participant behavior based on trading activity

    Args:
        market_data (Dict): Market data dictionary

    Returns:
        Dict[str, str]: Participant behavior analysis
    """
    # Placeholder function for analyzing participant behavior
    # Implement logic to assess market participant behavior based on trading activity
    # Consider factors like volume, order flow, sentiment, etc.

    behavior = {
        'retail_traders': 'Bullish',
        'institutional_investors': 'Neutral',
        'algorithmic_traders': 'Bearish'
    }

    return behavior

def determine_optimal_strategy(market_sentiment: str, participant_behavior: Dict[str, str]) -> str:
    """
    Determine the optimal trading strategy based on market sentiment and participant behavior

    Args:
        market_sentiment (str): Overall market sentiment
        participant_behavior (Dict[str, str]): Participant behavior analysis

    Returns:
        str: Optimal trading strategy
    """
    # Placeholder function for determining optimal strategy
    # Implement logic to determine the optimal trading strategy considering market sentiment and participant behavior
    # Apply game theory principles to identify the most advantageous approach

    if market_sentiment == 'Bullish' and participant_behavior['institutional_investors'] == 'Bullish':
        return 'Long'
    elif market_sentiment == 'Bearish' and participant_behavior['algorithmic_traders'] == 'Bearish':
        return 'Short'
    else:
        return 'Neutral'

def convert_numpy_types(obj):
    """
    Convert NumPy data types to standard Python types for JSON serialization.

    Args:
        obj: Input object that might contain NumPy types

    Returns:
        Converted object with standard Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def perform_market_analysis(payload_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive market analysis with enhanced validation"""
    try:
        # Validate payload first
        if not validate_payload(payload_data):
            raise ValueError("Invalid payload structure")

        # Initialize market analyzer and validator
        market_analyzer = MarketAnalyzer(payload_data)
        
        # Perform individual analyses
        market_structure = market_analyzer.analyze_market_structure()
        technical_analysis = analyze_technical_indicators(payload_data)
        
        # Validate and enhance options data
        options_data = analyze_options_chain(payload_data)
        validated_options = MarketDataValidator.validate_options_data(options_data)
        
        # Validate futures data
        futures_data = payload_data.get('futures', {})
        validated_futures = MarketDataValidator.validate_futures_data(
            futures_data, 
            payload_data.get('current_market', {}).get('index', {})
        )
        
        # Prepare comprehensive market data
        market_data = {
            'market_structure': market_structure,
            'technical_analysis': technical_analysis,
            'options_analysis': validated_options,
            'futures_analysis': validated_futures
        }
        
        # Enhance market signals
        enhanced_market_data = MarketDataValidator.enhance_market_signals(market_data)
        
        # Generate analysis results
        analysis_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'market_structure': market_structure,
            'options_analysis': validated_options,
            'technical_analysis': technical_analysis,
            'futures_analysis': validated_futures,
            'enhanced_signals': enhanced_market_data.get('enhanced_signals', {}),
            'summary': generate_market_summary(
                market_structure,
                validated_options,
                technical_analysis
            ),
            'trading_signals': generate_trading_signals(enhanced_market_data)
        }

        return convert_numpy_types(analysis_results)

    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }
    
def start_flask_app():
    """
    Start the Flask application
    """
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)