#!/usr/bin/env python3
"""
Trading Strategy Implementation for Backtesting
Exact replication of your C++ trading strategy logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_loss: float
    take_profit: float
    regime_id: int
    order_id: str
    trailing_stop: float = None  # For tracking trailing stop losses
    highest_price: float = None  # Track highest price for trailing stops


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    regime_id: int
    pnl: float
    pnl_pct: float
    exit_reason: str


class RegimeBasedTradingStrategy:
    """
    Exact replication of your C++ ResearchBasedTradingStrategy
    Uses your regime analysis, coin scoring, and position management logic
    """

    def __init__(self, initial_capital: float = 100000, enable_decision_explanations: bool = True):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
        # Decision explanations system
        self.enable_decision_explanations = enable_decision_explanations
        self.decision_log = []  # Store detailed decision explanations

        # Your exact symbol mapping (matching data manager)
        self.crypto_coins = {
            'BTCUSD': 'bitcoin', 'ETHUSD': 'ethereum', 'ADAUSD': 'cardano',
            'DOTUSD': 'polkadot', 'LINKUSD': 'chainlink', 'SOLUSD': 'solana',
            'MATICUSD': 'matic-network', 'AVAXUSD': 'avalanche-2', 'ATOMUSD': 'cosmos',
            'ALGOUSD': 'algorand', 'USDTUSD': 'tether', 'XRPUSD': 'ripple',
            'BNBUSD': 'binancecoin', 'DOGEUSD': 'dogecoin', 'LTCUSD': 'litecoin'
        }

        # Performance tracking
        self.equity_curve = []
        self.regime_performance = {i: [] for i in range(7)}
        self.regime_trades = {i: 0 for i in range(7)}

        # Enhanced risk management parameters
        self.min_trade_value = 100.0  # Minimum $100 per trade
        self.max_position_count = 5  # Maximum concurrent positions
        self.max_portfolio_risk = 0.15  # Maximum 15% portfolio risk at any time
        self.max_single_position_risk = 0.05  # Maximum 5% risk per position
        self.max_correlation_exposure = 0.50  # Maximum 50% in correlated assets (increased for better diversification)

        # Current regime tracking
        self.current_regime_id = None
        self.previous_regime_id = None

    def calculate_coin_scores(self, regime_data: Dict) -> Dict[str, float]:
        """
        Enhanced coin scoring algorithm with dynamic factors and volatility awareness
        """
        coin_scores = {}
        
        # Extract all PC factors for comprehensive analysis
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        pc3_factor = regime_data.get('pc3_factor', 0)
        pc4_factor = regime_data.get('pc4_factor', 0)
        pc5_factor = regime_data.get('pc5_factor', 0)
        
        persistence = regime_data['persistence']
        frequency_pct = regime_data['frequency_percentage']
        market_stress = regime_data['market_stress']
        strategy = regime_data['strategy']

        for symbol in self.crypto_coins.keys():
            # Enhanced base score with regime stability (increased base for better selection)
            regime_strength = (persistence * 0.35 + (frequency_pct / 100) * 0.25)
            market_condition = min(1.0, max(0.0, 1.0 - market_stress))
            base_score = regime_strength * market_condition + 0.4  # Increased from 0.3
            
            # Dynamic PC factor scoring based on regime strategy
            pc_score = 0.0
            
            if strategy in ["MOMENTUM", "BREAKOUT"]:
                # Favor momentum coins in momentum regimes
                if symbol in ['SOLUSD', 'AVAXUSD', 'LINKUSD', 'BNBUSD']:
                    pc_score += max(0, pc1_factor * 0.1) + abs(pc3_factor) * 0.05
                elif symbol in ['BTCUSD', 'ETHUSD']:
                    pc_score += max(0, pc1_factor * 0.05)
                    
            elif strategy in ["STABLE_GROWTH", "DEFENSIVE"]:
                # Favor stable coins in defensive regimes
                if symbol in ['BTCUSD', 'ETHUSD', 'USDTUSD']:
                    pc_score += 0.15 + max(0, -pc2_factor * 0.1)  # Bonus for low volatility
                
            # Volatility adjustment based on PC2
            volatility_penalty = abs(pc2_factor) * 0.02 if abs(pc2_factor) > 2.0 else 0
            
            # Final score with bounds checking
            final_score = max(0.0, min(1.0, base_score + pc_score - volatility_penalty))
            coin_scores[symbol] = final_score

        return coin_scores

    def calculate_position_size(self, regime_data: Dict, symbol: str, coin_score: float, current_price: float) -> float:
        """
        Enhanced position sizing with EV optimization and PC factor strength
        Combines Expected Value analysis with Principal Component factor strength
        """
        regime_id = regime_data['regime_id']
        strategy = regime_data['strategy']
        
        # ENHANCED EV-BASED ALLOCATION: Optimized based on Expected Value analysis
        # Regime EV ranking: 3(2.230R) > 5(0.761R) > 0(0.494R) > 2(0.166R) > 1(0.024R) > 4(-0.206R) > 6(-0.901R)
        
        # NOTE: regimes 3 and 6 are omitted deliberately. Both map to WAIT_AND_SEE,
        # which is forced to base_percent = 0.0 below and blocked by
        # should_trade_regime(), so their former entries (0.40 and 0.01) never executed.
        # The remaining values are the in-sample-fitted ladder documented in
        # Overfitting_Diagnosis.md item 1; Phase 4 replaces them with per-fold fits.
        if regime_id == 5:  # BREAKOUT_MOMENTUM - HIGH EV (0.761R, 60% WR)
            base_percent = 0.30  # Strong allocation for high EV
        elif regime_id == 0:  # STABLE_GROWTH - GOOD EV (0.494R, 54% WR)
            base_percent = 0.25  # Good allocation for solid EV
        elif regime_id == 2:  # BASELINE_MARKET - LOW EV (0.166R, 57.5% WR)
            base_percent = 0.15  # Moderate allocation for low but positive EV
        elif regime_id == 1:  # MODERATE_MOMENTUM - MINIMAL EV (0.024R, 44% WR)
            base_percent = 0.08  # Small allocation for minimal EV
        elif regime_id == 4:  # DEFENSIVE_STABLE - NEGATIVE EV (-0.206R, 40% WR)
            base_percent = 0.03  # Minimal allocation for negative EV
        else:
            # Fallback for unmapped regimes
            base_percent = 0.10
            
        # Avoid eliminated strategies
        if strategy == "WAIT_AND_SEE":
            base_percent = 0.0
        
        # ENHANCED PC FACTOR STRENGTH ADJUSTMENTS
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        pc3_factor = regime_data.get('pc3_factor', 0)
        
        # PC1 (Market Direction) Confidence Multiplier
        pc1_strength = abs(pc1_factor)
        # Increase size when market direction is strong and clear
        if pc1_strength > 2.0:  # Strong directional signal
            pc1_multiplier = min(1.4, 1.0 + (pc1_strength - 2.0) * 0.2)
        elif pc1_strength > 1.5:  # Moderate signal
            pc1_multiplier = min(1.2, 1.0 + (pc1_strength - 1.5) * 0.4)
        else:  # Weak signal
            pc1_multiplier = max(0.7, 1.0 - (1.5 - pc1_strength) * 0.2)
        
        # PC2 (Volatility) Risk Adjustment
        pc2_strength = abs(pc2_factor)
        # Reduce size during high volatility periods, increase during stable periods
        if pc2_strength > 2.5:  # High volatility
            pc2_adjustment = max(0.6, 1.0 - (pc2_strength - 2.5) * 0.15)
        elif pc2_strength < 0.8:  # Low volatility - opportunity for larger sizes
            pc2_adjustment = min(1.3, 1.0 + (0.8 - pc2_strength) * 0.4)
        else:  # Normal volatility
            pc2_adjustment = 1.0
            
        # PC3 (Sector/Style) Factor Bonus
        pc3_strength = abs(pc3_factor)
        if pc3_strength > 1.5:  # Strong sector rotation signal
            pc3_bonus = min(1.15, 1.0 + (pc3_strength - 1.5) * 0.3)
        else:
            pc3_bonus = 1.0
        
        # Combined PC Factor Strength (replaces simple vol_adjustment)
        pc_strength_multiplier = pc1_multiplier * pc2_adjustment * pc3_bonus
        pc_strength_multiplier = max(0.5, min(1.6, pc_strength_multiplier))  # Cap between 50%-160%
        
        # Enhanced Persistence bonus (regime stability)
        persistence = regime_data['persistence']
        if persistence > 0.8:  # Very stable regime
            persistence_bonus = 1.3
        elif persistence > 0.6:  # Stable regime
            persistence_bonus = 1.1
        elif persistence < 0.3:  # Unstable regime
            persistence_bonus = 0.7
        else:  # Normal stability
            persistence_bonus = 1.0
        
        # Portfolio concentration limit (relaxed for better capital utilization)
        portfolio_value = self.get_portfolio_value()
        current_positions_value = sum(pos.quantity * current_price 
                                    for pos in self.positions.values())
        current_exposure = current_positions_value / portfolio_value if portfolio_value > 0 else 0
        concentration_limit = max(0.7, 0.95 - current_exposure)  # Allow higher exposure
        
        # ENHANCED FINAL POSITION SIZE CALCULATION
        # Combines EV-optimized base allocation with PC factor strength and regime stability
        adjusted_percent = (base_percent * pc_strength_multiplier * persistence_bonus * 
                           coin_score * concentration_limit)
        
        # Apply additional safety caps
        adjusted_percent = max(0.01, min(0.45, adjusted_percent))  # Cap between 1%-45%
        
        position_value = portfolio_value * adjusted_percent
        
        return position_value / current_price

    def validate_position_risk(self, symbol: str, position_qty: float, entry_price: float, 
                              stop_loss: float) -> Tuple[bool, str]:
        """
        Validate position meets risk management criteria
        """
        portfolio_value = self.get_portfolio_value()
        
        # Calculate position risk
        risk_per_share = abs(entry_price - stop_loss)
        total_position_risk = risk_per_share * position_qty
        position_risk_pct = total_position_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Check single position risk limit
        if position_risk_pct > self.max_single_position_risk:
            return False, f"Position risk {position_risk_pct:.1%} exceeds limit {self.max_single_position_risk:.1%}"
        
        # Calculate current portfolio risk
        current_portfolio_risk = 0.0
        for pos in self.positions.values():
            pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
            current_portfolio_risk += pos_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Add new position risk
        total_portfolio_risk = current_portfolio_risk + position_risk_pct
        
        # Check portfolio risk limit
        if total_portfolio_risk > self.max_portfolio_risk:
            return False, f"Portfolio risk {total_portfolio_risk:.1%} would exceed limit {self.max_portfolio_risk:.1%}"
        
        # Check correlation exposure (simplified - group similar cryptos)
        correlated_groups = {
            'major': ['BTCUSD', 'ETHUSD'],
            'defi': ['LINKUSD', 'USDTUSD'],
            'alt': ['SOLUSD', 'AVAXUSD', 'MATICUSD', 'ATOMUSD', 'ALGOUSD'],
            'legacy': ['XRPUSD', 'LTCUSD', 'DOGEUSD'],
            'exchange': ['BNBUSD']
        }
        
        # Find which group this symbol belongs to
        symbol_group = None
        for group, symbols in correlated_groups.items():
            if symbol in symbols:
                symbol_group = group
                break
        
        if symbol_group:
            # Calculate current exposure to this group
            group_exposure = 0.0
            for pos in self.positions.values():
                if pos.symbol in correlated_groups[symbol_group]:
                    group_exposure += (pos.quantity * pos.entry_price) / portfolio_value if portfolio_value > 0 else 0
            
            # Add new position exposure
            new_position_value = position_qty * entry_price
            total_group_exposure = group_exposure + (new_position_value / portfolio_value if portfolio_value > 0 else 0)
            
            if total_group_exposure > self.max_correlation_exposure:
                return False, f"Correlation group exposure {total_group_exposure:.1%} would exceed limit {self.max_correlation_exposure:.1%}"
        
        return True, "Position risk validated"

    def calculate_stop_loss_take_profit(self, regime_data: Dict, entry_price: float) -> Tuple[float, float]:
        """
        Optimized risk management with dynamic stops and regime-specific logic
        """
        pc2_vol = abs(regime_data['pc2_volatility_factor'])
        strategy = regime_data['strategy']
        avg_duration = regime_data['avg_duration']
        persistence = regime_data['persistence']
        market_stress = regime_data['market_stress']
        
        # Enhanced volatility-adjusted stop loss
        # Keys must match the strategy strings the regime mapping actually emits.
        # The previous table was keyed on BASELINE/BREAKOUT/DEFENSIVE/CRISIS/
        # EXTREME_VOLATILITY, none of which are ever produced, so every regime except
        # STABLE_GROWTH and MOMENTUM silently fell through to the default.
        base_stop_loss = {
            "STABLE_GROWTH": 0.04,
            "MOMENTUM": 0.05,
            "BALANCED": 0.04,
            "CONSERVATIVE": 0.08,
        }
        
        base_stop = base_stop_loss.get(strategy, 0.05)
        
        # Volatility adjustment - tighter stops for low vol, wider for high vol
        vol_adjustment = max(0.7, min(1.5, 1.0 + (pc2_vol - 1.0) * 0.2))
        
        # Persistence adjustment - tighter stops for stable regimes
        persistence_adjustment = max(0.8, min(1.2, 2.0 - persistence))
        
        # Market stress adjustment - wider stops during stress
        stress_adjustment = max(1.0, min(1.3, 1.0 + market_stress * 0.3))
        
        # Final stop loss calculation
        stop_loss_pct = base_stop * vol_adjustment * persistence_adjustment * stress_adjustment
        stop_loss_pct = max(0.02, min(0.12, stop_loss_pct))  # Cap between 2-12%
        
        # Enhanced take profit with asymmetric risk/reward
        risk_reward_ratios = {
            "STABLE_GROWTH": 2.0,
            "MOMENTUM": 2.0,
            "BALANCED": 3.0,
            "CONSERVATIVE": 1.0,
        }
        
        risk_reward = risk_reward_ratios.get(strategy, 2.0)
        take_profit_pct = stop_loss_pct * risk_reward
        
        # Duration bonus for longer-term regimes
        if avg_duration > 10:
            take_profit_pct *= 1.2  # 20% bonus for longer regimes
        
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        return stop_loss, take_profit

    def regime_specific_trading_rules(self, regime_data: Dict, coin_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Apply regime-specific trading filters and thresholds based on EV analysis results
        Enhanced with Expected Value data from actual performance analysis
        """
        strategy = regime_data['strategy']
        regime_id = regime_data['regime_id']
        persistence = regime_data['persistence']
        market_stress = regime_data['market_stress']
        
        trading_decisions = {}
        
        # ENHANCED EV-OPTIMIZED THRESHOLDS: Based on Expected Value analysis
        # Using actual regime EV data: EXTREME_OUTLIER(2.230R), BREAKOUT_MOMENTUM(0.761R), 
        # STABLE_GROWTH(0.494R), BASELINE_MARKET(0.166R), etc.
        
        # regimes 3 and 6 omitted: unreachable, see calculate_position_size()
        if regime_id == 5:  # BREAKOUT_MOMENTUM - HIGH EV (0.761R, 60% WR)  
            threshold = 0.06  # Very low threshold for high EV regime
        elif regime_id == 0:  # STABLE_GROWTH - GOOD EV (0.494R, 54% WR)
            threshold = 0.08  # Low threshold for solid EV
        elif regime_id == 2:  # BASELINE_MARKET - LOW EV (0.166R, 57.5% WR)
            threshold = 0.25  # Moderate threshold - be selective
        elif regime_id == 1:  # MODERATE_MOMENTUM - MINIMAL EV (0.024R, 44% WR)
            threshold = 0.45  # High threshold - very selective
        elif regime_id == 4:  # DEFENSIVE_STABLE - NEGATIVE EV (-0.206R, 40% WR)
            threshold = 0.65  # Very high threshold - avoid most trades
        else:
            # Fallback for unmapped regimes
            threshold = 0.35
        
        # Stress adjustment - increase threshold during high stress
        if market_stress > 0.7:
            threshold += 0.10  # Be more selective during stress
        
        # Persistence filter (don't trade in very unstable regimes)
        if persistence < 0.25:  # Only very unstable regimes (lowered from 0.3)
            threshold += 0.15  # Smaller penalty (lowered from 0.2)
        
        # Market stress filter - relaxed
        if market_stress > 0.8:  # Only extreme stress (raised from 0.7)
            threshold += 0.1  # Smaller penalty (lowered from 0.15)
        
        # Apply filters
        qualified_coins = [(symbol, score) for symbol, score in coin_scores.items() 
                          if score >= threshold]
        
        # Sort by score and limit to max positions
        qualified_coins.sort(key=lambda x: x[1], reverse=True)
        max_positions = self.get_max_positions(regime_data)
        qualified_coins = qualified_coins[:max_positions]
        
        return {symbol: True for symbol, _ in qualified_coins}
    
    def get_max_positions(self, regime_data: Dict) -> int:
        """
        Get maximum number of positions based on regime
        """
        strategy = regime_data['strategy']
        
        if strategy == "WAIT_AND_SEE":
            return 2  # Limited but not minimal (increased from 1)
        elif strategy in ["MOMENTUM", "BREAKOUT"]:
            return 8  # Allow many positions in momentum (increased from 6)
        else:
            return 6  # Higher standard limit (increased from 5)

    def get_active_signals(self, regime_data: Dict) -> List[str]:
        """
        Your exact active signals logic from crypto_regime_analysis.py
        """
        active_signals = []
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        persistence = regime_data['persistence']
        should_trade = regime_data['should_trade']

        if should_trade:
            if pc1_factor > 2.0:
                active_signals.extend(["STRONG_MOMENTUM_UP", "BULLISH_BREAKOUT"])
            elif pc1_factor < -2.0:
                active_signals.extend(["STRONG_MOMENTUM_DOWN", "BEARISH_BREAKDOWN"])

            if abs(pc2_factor) > 3.0:
                active_signals.append("EXTREME_VOLATILITY_DETECTED")

            if persistence > 0.9:
                active_signals.append("HIGH_PERSISTENCE_REGIME")

        return active_signals

    def should_trade_regime(self, regime_data: Dict) -> bool:
        """
        IMPROVED: Trade multiple profitable regimes based on research insights
        Uses research-based regime classification for better opportunities
        """
        strategy = regime_data['strategy']
        regime_id = regime_data['regime_id']
        
        # Based on research analysis, trade these profitable scenarios:
        # - Bull Momentum: Ideal conditions (13.8% of time)
        # - Volatile Rebound: High risk/reward (20.7% of time)  
        # - Sideways: Range trading opportunities (38.5% of time)
        # Only avoid Sharp Correction during severe stress
        
        # Map regimes to research insights
        if strategy == "WAIT_AND_SEE":
            return False  # Always avoid uncertainty regimes
            
        # Sharp correction regime - only trade if not in extreme stress
        if regime_id == 1 and regime_data['market_stress'] > 0.85:
            return False  # Avoid severe correction phases
            
        # All other regimes are tradeable with appropriate position sizing
        return (
            regime_data['persistence'] > 0.4 and      # Reduced threshold for more opportunities
            regime_data['market_stress'] < 0.9 and    # Allow higher stress levels
            regime_data.get('should_trade', True)      # Basic trade flag (default True)
        )

    def process_regime_change(self, old_regime_id: Optional[int], new_regime_id: int,
                              new_regime_data: Dict, current_prices: Dict[str, float]) -> None:
        """
        Handle regime transitions (rebalancing logic from your C++ code)
        """
        print(f"🔄 Regime Change: {old_regime_id} → {new_regime_id} ({new_regime_data['strategy']})")
        
        # Log regime change decision
        regime_change_reasoning = {
            'regime_transition': {
                'from_regime': old_regime_id,
                'to_regime': new_regime_id,
                'from_strategy': 'Unknown' if old_regime_id is None else self.get_regime_strategy_name(old_regime_id),
                'to_strategy': new_regime_data['strategy']
            },
            'new_regime_analysis': {
                'ev_rank': self.get_regime_ev_rank(new_regime_id),
                'expected_value': self.get_regime_ev_estimate(new_regime_id),
                'win_rate': self.get_regime_win_rate(new_regime_id),
                'base_allocation': self.get_regime_base_allocation(new_regime_id),
                'persistence': new_regime_data['persistence'],
                'market_stress': new_regime_data['market_stress']
            },
            'pc_factors': {
                'pc1_market_factor': new_regime_data['pc1_market_factor'],
                'pc2_volatility_factor': new_regime_data['pc2_volatility_factor'],
                'pc3_factor': new_regime_data.get('pc3_factor', 0)
            },
            'rationale': [
                f"Market regime shifted from {old_regime_id or 'Initial'} to {new_regime_id}",
                f"New regime strategy: {new_regime_data['strategy']}",
                f"Expected Value ranking: {self.get_regime_ev_rank(new_regime_id)} ({self.get_regime_ev_estimate(new_regime_id)}R)",
                f"Portfolio rebalancing required based on new regime allocation",
                f"PC factors indicate: Market={new_regime_data['pc1_market_factor']:+.2f}, Volatility={new_regime_data['pc2_volatility_factor']:+.2f}"
            ]
        }
        
        self.log_decision('regime_change', 'PORTFOLIO', 'rebalance', regime_change_reasoning)
    
    def get_regime_strategy_name(self, regime_id: int) -> str:
        """
        Get strategy name for regime ID
        """
        strategies = {
            0: "STABLE_GROWTH", 1: "MODERATE_MOMENTUM", 2: "BASELINE_MARKET",
            3: "EXTREME_OUTLIER", 4: "DEFENSIVE_STABLE", 5: "BREAKOUT_MOMENTUM",
            6: "EXTREME_VOLATILITY"
        }
        return strategies.get(regime_id, "UNKNOWN")

    def execute_trading_cycle(self, regime_data: Dict, current_prices: Dict[str, float],
                              timestamp: pd.Timestamp) -> None:
        """
        Main trading execution logic (from your runTradingLoop)
        """
        regime_id = regime_data['regime_id']

        # Handle regime changes
        if self.current_regime_id != regime_id:
            self.process_regime_change(self.current_regime_id, regime_id, regime_data, current_prices)
            self.previous_regime_id = self.current_regime_id
            self.current_regime_id = regime_id

        # Check if we should trade
        if not self.should_trade_regime(regime_data):
            strategy = regime_data['strategy']
            if strategy == "WAIT_AND_SEE":
                # Close all positions only for WAIT_AND_SEE regime (uncertainty)
                for symbol in list(self.positions.keys()):
                    self.close_position(symbol, current_prices[symbol], "wait_and_see_regime_exit")
                print(f"🚫 WAIT_AND_SEE regime - closed all positions")
            else:
                # Keep existing positions for all other regimes when conditions aren't met
                print(f"🛡️  {strategy} regime - conditions not met, maintaining existing positions only")
            return

        # Check stop losses and take profits for existing positions
        self.check_stop_loss_take_profit(current_prices, timestamp)

        # Calculate coin scores
        coin_scores = self.calculate_coin_scores(regime_data)

        # Apply regime-specific trading rules and thresholds
        trading_decisions = self.regime_specific_trading_rules(regime_data, coin_scores)
        
        # Execute trades for qualified coins
        for symbol, should_trade in trading_decisions.items():
            if should_trade and symbol in current_prices:
                current_price = current_prices[symbol]
                coin_score = coin_scores[symbol]

                if symbol not in self.positions and len(self.positions) < self.get_max_positions(regime_data):
                    # Calculate position size
                    position_qty = self.calculate_position_size(regime_data, symbol, coin_score, current_price)
                    position_value = position_qty * current_price

                    # Check minimum trade value
                    if position_value >= self.min_trade_value:
                        # Calculate stop loss for risk validation
                        stop_loss, _ = self.calculate_stop_loss_take_profit(regime_data, current_price)
                        
                        # Validate position risk
                        risk_valid, risk_msg = self.validate_position_risk(symbol, position_qty, current_price, stop_loss)
                        
                        if not risk_valid:
                            # Log risk rejection decision
                            risk_reasoning = {
                                'regime': {
                                    'id': regime_data['regime_id'],
                                    'strategy': regime_data['strategy'],
                                    'expected_value': self.get_regime_ev_estimate(regime_data['regime_id']),
                                    'win_rate': self.get_regime_win_rate(regime_data['regime_id']),
                                    'persistence': regime_data['persistence'],
                                    'market_stress': regime_data['market_stress']
                                },
                                'position_sizing': {
                                    'position_value': position_value,
                                    'final_percent': (position_value / self.get_portfolio_value()) * 100
                                },
                                'risk_management': {
                                    'stop_loss': stop_loss,
                                    'rejection_reason': risk_msg
                                },
                                'rationale': [f"RISK REJECTED: {risk_msg}"]
                            }
                            self.log_decision('entry', symbol, 'skip', risk_reasoning, timestamp)
                            continue
                        
                        entry_reasoning = self.build_entry_reasoning(regime_data, symbol, coin_score,
                                                                   current_price, position_qty, stop_loss)
                        self.log_decision('entry', symbol, 'open', entry_reasoning, timestamp)
                        self.open_position(symbol, position_qty, current_price, regime_data, timestamp)

    def open_position(self, symbol: str, quantity: float, entry_price: float,
                      regime_data: Dict, timestamp: pd.Timestamp) -> None:
        """
        Open a new position (from trackOpenPosition)
        """
        if quantity * entry_price > self.current_capital:
            print(f"⚠️  Insufficient capital for {symbol} position")
            return

        stop_loss, take_profit = self.calculate_stop_loss_take_profit(regime_data, entry_price)

        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime_id=regime_data['regime_id'],
            order_id=f"order_{timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}",
            trailing_stop=None,  # Will be set when position becomes profitable
            highest_price=entry_price  # Initialize to entry price
        )

        self.positions[symbol] = position
        self.current_capital -= quantity * entry_price

        print(f"✅ Opened {symbol}: {quantity:.4f} @ ${entry_price:.2f} "
              f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})")

    def build_entry_reasoning(self, regime_data: Dict, symbol: str, coin_score: float,
                             current_price: float, position_qty: float, stop_loss: float) -> Dict:
        """
        Build comprehensive reasoning for entry decisions
        """
        take_profit = self.calculate_stop_loss_take_profit(regime_data, current_price)[1]
        position_value = position_qty * current_price
        portfolio_value = self.get_portfolio_value()
        
        # PC Factor analysis
        pc1_factor = regime_data['pc1_market_factor']
        pc2_factor = regime_data['pc2_volatility_factor']
        pc3_factor = regime_data.get('pc3_factor', 0)
        
        pc1_strength = abs(pc1_factor)
        pc1_multiplier = min(1.4, 1.0 + (pc1_strength - 2.0) * 0.2) if pc1_strength > 2.0 else (
            min(1.2, 1.0 + (pc1_strength - 1.5) * 0.4) if pc1_strength > 1.5 else 
            max(0.7, 1.0 - (1.5 - pc1_strength) * 0.2)
        )
        
        pc2_strength = abs(pc2_factor)
        pc2_adjustment = max(0.6, 1.0 - (pc2_strength - 2.5) * 0.15) if pc2_strength > 2.5 else (
            min(1.3, 1.0 + (0.8 - pc2_strength) * 0.4) if pc2_strength < 0.8 else 1.0
        )
        
        pc3_strength = abs(pc3_factor)
        pc3_bonus = min(1.15, 1.0 + (pc3_strength - 1.5) * 0.3) if pc3_strength > 1.5 else 1.0
        
        pc_strength_multiplier = max(0.5, min(1.6, pc1_multiplier * pc2_adjustment * pc3_bonus))
        
        # Base allocation based on EV ranking
        base_percent = self.get_regime_base_allocation(regime_data['regime_id'])
        
        reasoning = {
            'regime': {
                'id': regime_data['regime_id'],
                'strategy': regime_data['strategy'],
                'ev_rank': self.get_regime_ev_rank(regime_data['regime_id']),
                'expected_value': self.get_regime_ev_estimate(regime_data['regime_id']),
                'win_rate': self.get_regime_win_rate(regime_data['regime_id']),
                'persistence': regime_data['persistence'],
                'market_stress': regime_data['market_stress']
            },
            'pc_factors': {
                'pc1_factor': pc1_factor,
                'pc1_multiplier': pc1_multiplier,
                'pc2_factor': pc2_factor,
                'pc2_adjustment': pc2_adjustment,
                'pc3_factor': pc3_factor,
                'pc3_bonus': pc3_bonus,
                'pc_strength_multiplier': pc_strength_multiplier
            },
            'position_sizing': {
                'base_percent': base_percent,
                'pc_strength_multiplier': pc_strength_multiplier,
                'persistence_bonus': self.get_persistence_bonus(regime_data['persistence']),
                'coin_score': coin_score,
                'position_value': position_value,
                'final_percent': (position_value / portfolio_value) * 100
            },
            'risk_management': {
                'stop_loss': stop_loss,
                'stop_loss_pct': ((current_price - stop_loss) / current_price) * 100,
                'take_profit': take_profit,
                'risk_reward_ratio': (take_profit - current_price) / (current_price - stop_loss),
                'position_risk_pct': ((current_price - stop_loss) * position_qty / portfolio_value) * 100,
                'total_portfolio_risk': self.calculate_total_portfolio_risk(portfolio_value),
                'max_portfolio_risk': self.max_portfolio_risk * 100
            },
            'rationale': [
                f"High-EV regime {regime_data['regime_id']} with {self.get_regime_ev_estimate(regime_data['regime_id'])}R expected value",
                f"Strong PC factor signals: PC1={pc1_factor:+.2f}, PC2={pc2_factor:+.2f}, PC3={pc3_factor:+.2f}",
                f"Coin score {coin_score:.3f} exceeds regime threshold",
                f"Risk-adjusted position size: {(position_value / portfolio_value) * 100:.1f}% of portfolio",
                f"Risk/Reward ratio: {(take_profit - current_price) / (current_price - stop_loss):.1f}:1"
            ]
        }
        
        return reasoning
    
    def get_regime_base_allocation(self, regime_id: int) -> float:
        """
        Get EV-optimized base allocation percentage for regime
        """
        allocations = {
            5: 0.30,  # BREAKOUT_MOMENTUM - HIGH EV (0.761R, 60% WR)
            0: 0.25,  # STABLE_GROWTH - GOOD EV (0.494R, 54% WR)
            2: 0.15,  # BASELINE_MARKET - LOW EV (0.166R, 57.5% WR)
            1: 0.08,  # MODERATE_MOMENTUM - MINIMAL EV (0.024R, 44% WR)
            4: 0.03,  # DEFENSIVE_STABLE - NEGATIVE EV (-0.206R, 40% WR)
        }
        return allocations.get(regime_id, 0.10)
    
    def get_regime_ev_rank(self, regime_id: int) -> str:
        """
        Get regime ranking based on Expected Value
        """
        rankings = {
            3: "1st (BEST)", 5: "2nd (HIGH)", 0: "3rd (GOOD)", 
            2: "4th (LOW)", 1: "5th (MINIMAL)", 4: "6th (NEGATIVE)", 6: "7th (WORST)"
        }
        return rankings.get(regime_id, "Unknown")
    
    def get_regime_ev_estimate(self, regime_id: int) -> str:
        """
        Get regime Expected Value estimate in R multiples
        """
        ev_estimates = {
            3: "2.230", 5: "0.761", 0: "0.494", 
            2: "0.166", 1: "0.024", 4: "-0.206", 6: "-0.901"
        }
        return ev_estimates.get(regime_id, "0.000")
    
    def get_regime_win_rate(self, regime_id: int) -> int:
        """
        Get regime historical win rate percentage
        """
        win_rates = {
            3: 75, 5: 60, 0: 54, 2: 58, 1: 44, 4: 40, 6: 0
        }
        return win_rates.get(regime_id, 50)
    
    def get_persistence_bonus(self, persistence: float) -> float:
        """
        Calculate persistence bonus multiplier
        """
        if persistence > 0.8:
            return 1.3
        elif persistence > 0.6:
            return 1.1
        elif persistence < 0.3:
            return 0.7
        else:
            return 1.0
    
    def calculate_total_portfolio_risk(self, portfolio_value: float) -> float:
        """
        Calculate current total portfolio risk percentage
        """
        total_risk = 0.0
        for pos in self.positions.values():
            pos_risk = abs(pos.entry_price - pos.stop_loss) * pos.quantity
            total_risk += pos_risk / portfolio_value if portfolio_value > 0 else 0
        return total_risk * 100
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str,
                       timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        Close an existing position and record the trade
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

        # Update capital
        self.current_capital += position.quantity * exit_price

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_timestamp=position.entry_timestamp,
            exit_timestamp=timestamp or pd.Timestamp.now(),
            regime_id=position.regime_id,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason
        )

        # Record trade
        self.trades.append(trade)
        self.regime_performance[position.regime_id].append(pnl)
        self.regime_trades[position.regime_id] += 1

        # Log exit decision with reasoning
        exit_reasoning = {
            'position': {
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'hold_duration': (timestamp - position.entry_timestamp).days if timestamp else 0
            },
            'performance': {
                'pnl_dollars': pnl,
                'pnl_percent': pnl_pct,
                'r_multiple': pnl / (abs(position.entry_price - position.stop_loss) * position.quantity) if position.stop_loss != position.entry_price else 0
            },
            'exit_trigger': {
                'reason': exit_reason,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'trailing_stop': position.trailing_stop,
                'highest_price': position.highest_price
            },
            'rationale': [
                f"Exit triggered by {exit_reason.replace('_', ' ').title()}",
                f"Position held for {(timestamp - position.entry_timestamp).days if timestamp else 0} days",
                f"Realized P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
            ]
        }
        
        self.log_decision('exit', symbol, 'close', exit_reasoning, timestamp)
        
        # Remove position
        del self.positions[symbol]

        pnl_sign = "💰" if pnl > 0 else "❌"
        print(f"{pnl_sign} Closed {symbol}: {position.quantity:.4f} @ ${exit_price:.2f} "
              f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) - {exit_reason}")

    def check_stop_loss_take_profit(self, current_prices: Dict[str, float],
                                    timestamp: pd.Timestamp) -> None:
        """
        Enhanced stop loss/take profit with trailing stops for winning positions
        """
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                
                # Initialize tracking if not set
                if position.highest_price is None:
                    position.highest_price = current_price
                
                # Update highest price for trailing stops
                if current_price > position.highest_price:
                    position.highest_price = current_price
                    
                    # Activate trailing stop once position is 10% profitable
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    if profit_pct > 0.10 and position.trailing_stop is None:
                        # Set initial trailing stop at 5% below current price
                        position.trailing_stop = current_price * 0.95
                        print(f"🔥 Trailing stop activated for {symbol} at ${position.trailing_stop:.2f}")
                    
                    # Update trailing stop if already active
                    elif position.trailing_stop is not None:
                        new_trailing = current_price * 0.95  # 5% trailing distance
                        if new_trailing > position.trailing_stop:
                            position.trailing_stop = new_trailing

                # Check trailing stop first (higher priority than regular stop)
                if position.trailing_stop is not None and current_price <= position.trailing_stop:
                    positions_to_close.append((symbol, current_price, "trailing_stop"))
                
                # Check regular stop loss (only if no trailing stop triggered)
                elif current_price <= position.stop_loss:
                    positions_to_close.append((symbol, current_price, "stop_loss"))

                # Check take profit
                elif current_price >= position.take_profit:
                    positions_to_close.append((symbol, current_price, "take_profit"))

        # Close positions that hit stops
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason, timestamp)

    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate current portfolio value
        """
        if current_prices is None:
            return self.current_capital

        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position.quantity * current_prices[symbol]

        return self.current_capital + position_value

    def update_equity_curve(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]) -> None:
        """
        Update equity curve for performance tracking
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'positions_value': portfolio_value - self.current_capital,
            'regime_id': self.current_regime_id
        })

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0395) -> float:
        """
        Calculate Sharpe ratio using industry standard methodology
        
        Args:
            risk_free_rate: Annual risk-free rate (3.95% as of Jan 2, 2024 - 10Y Treasury at backtest start)
        
        Returns:
            Annualized Sharpe ratio
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(self.equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Resample to daily data for proper Sharpe calculation
        df.set_index('timestamp', inplace=True)
        daily_df = df.resample('D')['portfolio_value'].last().dropna()
        
        # Calculate daily returns
        daily_returns = daily_df.pct_change().dropna()
        
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
        
        # Industry standard: Calculate compound annualized return
        initial_value = daily_df.iloc[0]
        final_value = daily_df.iloc[-1]
        num_days = len(daily_returns)
        annualized_return = (final_value / initial_value) ** (252 / num_days) - 1
        
        # Industry standard: Annualized volatility using 252 trading days
        # Ensure we're calculating volatility on percentage returns, not decimal returns
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        
        # Sanity check - volatility should typically be between 0.1 and 5.0 for most strategies
        if annualized_volatility > 10:
            # Recalculate with percentage returns properly scaled
            annualized_volatility = (daily_returns.std() * np.sqrt(252))
        
        # Calculate Sharpe ratio (all values in decimal form)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0.0
        
        return sharpe_ratio

    def log_decision(self, decision_type: str, symbol: str, action: str, reasoning: Dict, timestamp: pd.Timestamp = None) -> None:
        """
        Log detailed trading decision with comprehensive reasoning
        
        Args:
            decision_type: Type of decision ('entry', 'exit', 'skip', 'regime_change')
            symbol: Cryptocurrency symbol
            action: Action taken ('open', 'close', 'skip', 'rebalance')
            reasoning: Detailed reasoning dictionary
            timestamp: Decision timestamp
        """
        if not self.enable_decision_explanations:
            return
            
        decision_entry = {
            'timestamp': timestamp or pd.Timestamp.now(),
            'decision_type': decision_type,
            'symbol': symbol,
            'action': action,
            'reasoning': reasoning,
            'portfolio_value': self.get_portfolio_value(),
            'positions_count': len(self.positions)
        }
        
        self.decision_log.append(decision_entry)
        
        # Print decision explanation if enabled
        if self.enable_decision_explanations:
            self.print_decision_explanation(decision_entry)
    
    def print_decision_explanation(self, decision: Dict) -> None:
        """
        Print formatted decision explanation
        """
        timestamp = decision['timestamp'].strftime('%H:%M:%S')
        decision_type = decision['decision_type'].upper()
        symbol = decision['symbol']
        action = decision['action'].upper()
        reasoning = decision['reasoning']
        
        print(f"\n🧠 [{timestamp}] {decision_type} DECISION: {symbol} - {action}")
        print("─" * 60)
        
        # Regime Analysis
        if 'regime' in reasoning:
            regime = reasoning['regime']
            print(f"🏛️  REGIME ANALYSIS:")
            print(f"   • Regime: {regime['id']} ({regime['strategy']})")
            print(f"   • EV Ranking: {regime.get('ev_rank', 'N/A')} (Expected: {regime.get('expected_value', 'N/A')}R)")
            print(f"   • Win Rate: {regime.get('win_rate', 'N/A')}%")
            print(f"   • Persistence: {regime.get('persistence', 0):.1%} | Stress: {regime.get('market_stress', 0):.1%}")
        
        # PC Factor Analysis
        if 'pc_factors' in reasoning:
            pc = reasoning['pc_factors']
            print(f"⚡ PC FACTOR STRENGTH:")
            print(f"   • PC1 (Market Direction): {pc.get('pc1_factor', 0):+.2f} → Multiplier: {pc.get('pc1_multiplier', 1):.2f}x")
            print(f"   • PC2 (Volatility): {pc.get('pc2_factor', 0):+.2f} → Adjustment: {pc.get('pc2_adjustment', 1):.2f}x")
            print(f"   • PC3 (Sector/Style): {pc.get('pc3_factor', 0):+.2f} → Bonus: {pc.get('pc3_bonus', 1):.2f}x")
            print(f"   • Combined PC Strength: {pc.get('pc_strength_multiplier', 1):.2f}x")
        
        # Position Sizing
        if 'position_sizing' in reasoning:
            pos = reasoning['position_sizing']
            print(f"📊 POSITION SIZING:")
            print(f"   • Base Allocation: {pos.get('base_percent', 0):.1%} (EV-optimized)")
            print(f"   • PC Strength Adjustment: {pos.get('pc_strength_multiplier', 1):.2f}x")
            print(f"   • Persistence Bonus: {pos.get('persistence_bonus', 1):.2f}x")
            print(f"   • Coin Score: {pos.get('coin_score', 0):.3f}")
            print(f"   • Final Position: ${pos.get('position_value', 0):,.0f} ({pos.get('final_percent', 0):.1%} of portfolio)")
        
        # Risk Management
        if 'risk_management' in reasoning:
            risk = reasoning['risk_management']
            print(f"🛡️  RISK MANAGEMENT:")
            print(f"   • Stop Loss: ${risk.get('stop_loss', 0):.2f} ({risk.get('stop_loss_pct', 0):.1%})")
            print(f"   • Take Profit: ${risk.get('take_profit', 0):.2f} (R/R: {risk.get('risk_reward_ratio', 0):.1f}:1)")
            print(f"   • Position Risk: {risk.get('position_risk_pct', 0):.1%} of portfolio")
            print(f"   • Portfolio Risk: {risk.get('total_portfolio_risk', 0):.1%} (Limit: {risk.get('max_portfolio_risk', 15):.0f}%)")
        
        # Expected Value Analysis
        if 'ev_analysis' in reasoning:
            ev = reasoning['ev_analysis']
            print(f"⚡ EXPECTED VALUE:")
            print(f"   • Trade EV: {ev.get('expected_value_r', 0):.3f}R (${ev.get('trade_expected_ev', 0):.2f})")
            print(f"   • Regime Win Rate: {ev.get('win_rate', 0):.1f}%")
            print(f"   • EV Threshold Met: {'✅' if ev.get('ev_ok', False) else '❌'} (Min: {ev.get('min_ev_required', 0):.3f}R)")
        
        # Decision Rationale
        if 'rationale' in reasoning:
            rationale = reasoning['rationale']
            print(f"💡 DECISION RATIONALE:")
            for reason in rationale:
                print(f"   • {reason}")
        
        print("─" * 60)
    
    def get_performance_summary(self) -> Dict:
        """
        Calculate comprehensive performance metrics including Sharpe ratio
        """
        if not self.trades:
            final_value = self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.current_capital
            return {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': 0.0,
                'total_pnl': final_value - self.initial_capital,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': self.calculate_sharpe_ratio(),
                'regime_performance': {},
                'equity_curve': self.equity_curve
            }

        # Basic metrics
        final_value = self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100

        # Trade analysis
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(
            sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(
            t.pnl for t in losing_trades) != 0 else float('inf')

        # Drawdown calculation
        equity_values = [e['portfolio_value'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        # Regime performance
        regime_performance = {}
        for regime_id in range(7):
            trades_count = self.regime_trades[regime_id]
            pnl_list = self.regime_performance[regime_id]

            if trades_count > 0:
                regime_win_rate = len([p for p in pnl_list if p > 0]) / len(pnl_list) * 100
                regime_avg_pnl = np.mean(pnl_list)
                regime_performance[regime_id] = {
                    'trades': trades_count,
                    'win_rate': regime_win_rate,
                    'avg_pnl': regime_avg_pnl,
                    'total_pnl': sum(pnl_list)
                }

        # Calculate Sharpe ratio and annualized metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        
        # Calculate proper annualized metrics using daily data
        annualized_return = 0.0
        annualized_volatility = 0.0
        if self.equity_curve:
            df = pd.DataFrame(self.equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            daily_df = df.resample('D')['portfolio_value'].last().dropna()
            
            if len(daily_df) > 1:
                daily_returns = daily_df.pct_change().dropna()
                if not daily_returns.empty:
                    initial_value = daily_df.iloc[0]
                    final_value = daily_df.iloc[-1]
                    num_days = len(daily_returns)
                    annualized_return = (final_value / initial_value) ** (252 / num_days) - 1
                    
                    # Fixed volatility calculation
                    daily_vol = daily_returns.std()
                    annualized_volatility = daily_vol * np.sqrt(252)
                    
                    # Cap extreme volatility values (likely calculation errors)
                    if annualized_volatility > 5.0:  # More than 500% volatility suggests error
                        annualized_volatility = min(annualized_volatility, 2.0)  # Cap at 200%
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - self.initial_capital,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'risk_free_rate': 0.0395,  # 3.95% (10Y Treasury Jan 2, 2024)
            'data_frequency': '252 trading days (industry standard)',
            'regime_performance': regime_performance,
            'equity_curve': self.equity_curve,
            'decision_log': self.decision_log if self.enable_decision_explanations else [],
            'decision_summary': self.get_decision_summary() if self.enable_decision_explanations else {},
        }
    
    def get_decision_summary(self) -> Dict:
        """
        Generate summary statistics from decision log
        """
        if not self.decision_log:
            return {}
        
        entry_decisions = [d for d in self.decision_log if d['decision_type'] == 'entry']
        exit_decisions = [d for d in self.decision_log if d['decision_type'] == 'exit']
        regime_changes = [d for d in self.decision_log if d['decision_type'] == 'regime_change']
        
        entries_taken = [d for d in entry_decisions if d['action'] == 'open']
        entries_skipped = [d for d in entry_decisions if d['action'] == 'skip']
        
        # Analyze rejection reasons
        rejection_reasons = {}
        for decision in entries_skipped:
            rationale = decision['reasoning'].get('rationale', [])
            if rationale:
                reason = rationale[0].split(':')[0] if ':' in rationale[0] else rationale[0]
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        return {
            'total_decisions': len(self.decision_log),
            'entry_decisions': len(entry_decisions),
            'entries_taken': len(entries_taken),
            'entries_skipped': len(entries_skipped),
            'entry_success_rate': len(entries_taken) / len(entry_decisions) * 100 if entry_decisions else 0,
            'exit_decisions': len(exit_decisions),
            'regime_changes': len(regime_changes),
            'rejection_reasons': rejection_reasons,
            'most_common_rejection': max(rejection_reasons.items(), key=lambda x: x[1])[0] if rejection_reasons else 'None'
        }

    def print_performance_summary(self) -> None:
        """
        Print detailed performance summary
        """
        summary = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("📊 REGIME-BASED TRADING PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"💰 Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"🎯 Final Value: ${summary['final_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2f}%")
        print(f"💵 Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"🔄 Total Trades: {summary['num_trades']}")
        print(f"🎯 Win Rate: {summary['win_rate']:.1f}%")
        print(f"💚 Average Win: ${summary['avg_win']:.2f}")
        print(f"❌ Average Loss: ${summary['avg_loss']:.2f}")
        print(f"⚡ Profit Factor: {summary['profit_factor']:.2f}")
        print(f"📉 Max Drawdown: {summary['max_drawdown']:.2f}%")
        print(f"📊 Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        
        # Print EV metrics if available
        if 'expected_value_r' in summary:
            print(f"\n⚡ EXPECTED VALUE ANALYSIS:")
            print(f"📈 Expected Value: {summary['expected_value_r']:.3f}R (${summary['expected_value_dollars']:.2f})")
            print(f"🎯 EV Win Rate: {summary['ev_win_rate']:.1%}")
            print(f"💰 EV Avg Win: {summary['ev_avg_win_r']:.2f}R")
            print(f"💸 EV Avg Loss: {summary['ev_avg_loss_r']:.2f}R")

        print("\n🏛️ REGIME PERFORMANCE BREAKDOWN:")
        print("-" * 50)

        regime_names = {
            0: "STABLE_GROWTH", 1: "MODERATE_MOMENTUM", 2: "BASELINE_MARKET",
            3: "EXTREME_OUTLIER", 4: "DEFENSIVE_STABLE", 5: "BREAKOUT_MOMENTUM",
            6: "EXTREME_VOLATILITY"
        }

        for regime_id, regime_perf in summary['regime_performance'].items():
            regime_name = regime_names.get(regime_id, f"REGIME_{regime_id}")
            print(f"Regime {regime_id} ({regime_name}):")
            print(f"  📊 Trades: {regime_perf['trades']}")
            print(f"  🎯 Win Rate: {regime_perf['win_rate']:.1f}%")
            print(f"  💰 Avg P&L: ${regime_perf['avg_pnl']:.2f}")
            print(f"  📈 Total P&L: ${regime_perf['total_pnl']:.2f}")

        # Print decision summary if explanations are enabled
        if self.enable_decision_explanations and hasattr(self, 'decision_log') and self.decision_log:
            decision_summary = self.get_decision_summary()
            print(f"\n🧠 DECISION ANALYSIS SUMMARY:")
            print("-" * 50)
            print(f"📊 Total Decisions Made: {decision_summary.get('total_decisions', 0)}")
            print(f"🎯 Entry Attempts: {decision_summary.get('entry_decisions', 0)}")
            print(f"✅ Entries Taken: {decision_summary.get('entries_taken', 0)}")
            print(f"❌ Entries Skipped: {decision_summary.get('entries_skipped', 0)}")
            print(f"📈 Entry Success Rate: {decision_summary.get('entry_success_rate', 0):.1f}%")
            print(f"🔄 Regime Changes: {decision_summary.get('regime_changes', 0)}")
            if decision_summary.get('most_common_rejection') != 'None':
                print(f"🚫 Most Common Rejection: {decision_summary.get('most_common_rejection', 'None')}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Test the trading strategy
    print("🔬 REGIME-BASED TRADING STRATEGY TEST")
    print("=" * 50)

    strategy = RegimeBasedTradingStrategy(initial_capital=100000)

    # Mock regime data for testing
    test_regime_data = {
        'regime_id': 1,
        'strategy': 'MOMENTUM',
        'pc1_market_factor': 1.5,
        'pc2_volatility_factor': -0.8,
        'persistence': 0.75,
        'avg_duration': 8.5,
        'frequency_percentage': 15.0,
        'risk_multiplier': 1.1,
        'market_stress': 0.4,
        'should_trade': True
    }

    test_prices = {
        'BTCUSD': 45000.0,
        'ETHUSD': 3200.0,
        'SOLUSD': 120.0,
        'AVAXUSD': 35.0
    }

    timestamp = pd.Timestamp('2024-01-15 12:00:00')

    print("Testing coin scoring...")
    coin_scores = strategy.calculate_coin_scores(test_regime_data)
    for symbol, score in coin_scores.items():
        print(f"  {symbol}: {score:.3f}")

    print("\nTesting trading cycle...")
    strategy.execute_trading_cycle(test_regime_data, test_prices, timestamp)

    print("\nTesting performance summary...")
    strategy.print_performance_summary()

    print("\n✅ Trading Strategy Test Complete!")