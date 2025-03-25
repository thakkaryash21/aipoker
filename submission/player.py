from agents.agent import Agent
from gym_env import PokerEnv
import numpy as np
import random
import time
from collections import defaultdict
import pickle
import os
import math

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card
int_card_to_str = PokerEnv.int_card_to_str


class PlayerAgent(Agent):
    def __name__(self):
        return "CFR-HybridAgent"

    def __init__(self, stream: bool = True, player_id: str = None):
        super().__init__(stream, player_id)
        self.hand_start_time = 0
        self.hand_number = 0
        self.street_history = []
        self.actions_this_hand = []
        self.discard_performed = False
        self.time_used_per_hand = []
        self.max_time_per_hand = 0.5  # Initial conservative limit

        # Initialize CFR strategy data structures
        self.regret_sum = defaultdict(
            lambda: np.zeros(5)
        )  # Maps infosets to regret sums
        self.strategy_sum = defaultdict(
            lambda: np.zeros(5)
        )  # Maps infosets to strategy sums
        self.iterations = 0

        # Card abstraction
        self.card_clusters = self._create_card_abstraction()

        # Pre-flop strategy
        self.preflop_strategy = self._initialize_preflop_strategy()

        # Opponent modeling components
        self.opp_action_counts = defaultdict(
            lambda: np.zeros(5)
        )  # Action counts by street
        self.opp_betting_frequency = defaultdict(
            float
        )  # Raise/call frequencies by street
        self.opp_fold_frequency = defaultdict(float)  # Fold frequencies by street
        self.opp_discard_frequency = 0.0  # How often opponent discards
        self.total_observed_hands = 0

        # Exploitation parameters
        self.exploitation_factor = 0.0  # Start with unexploitable strategy
        self.confidence_threshold = 30  # Number of hands before starting exploitation
        self.aggression_factor = 1.0  # Default aggression factor
        self.discard_threshold = 0.0  # Default discard threshold

        # Discard strategy components
        self.discard_ev_table = self._initialize_discard_ev_table()

        # Load pre-computed strategies and verify their quality
        self._load_strategy_files()

        # Hand evaluator
        self.hand_evaluator = HandEvaluator()

    def _create_card_abstraction(self):
        """Create more detailed card clusters for abstraction."""
        clusters = {}

        # Improved clusters based on rank and suit properties
        for card in range(27):
            rank = card % 9
            suit = card // 9

            if rank == 8:  # Aces
                clusters[card] = 4  # Highest cluster
            elif rank >= 6:  # 8-9
                clusters[card] = 3  # High cards
            elif rank >= 4:  # 6-7
                clusters[card] = 2  # Medium cards
            elif rank >= 2:  # 4-5
                clusters[card] = 1  # Low-medium cards
            else:  # 2-3
                clusters[card] = 0  # Low cards

        return clusters

    def _initialize_preflop_strategy(self):
        """Initialize pre-computed preflop strategy."""
        # Create a map of starting hands to strategies
        strategy = {}

        # Pairs have higher raising percentages
        for rank in range(9):  # 2-A
            for i in range(3):  # 3 suits
                for j in range(i + 1, 3):
                    pair_hand = tuple(sorted([rank + i * 9, rank + j * 9]))
                    if rank == 8:  # Aces
                        strategy[pair_hand] = [0.0, 0.7, 0.0, 0.3, 0.0]  # Aggressive
                    elif rank >= 6:  # High pairs (8-9)
                        strategy[pair_hand] = [0.0, 0.5, 0.1, 0.4, 0.0]  # Strong
                    elif rank >= 4:  # Medium pairs (6-7)
                        strategy[pair_hand] = [0.1, 0.3, 0.2, 0.4, 0.0]  # Moderate
                    else:  # Low pairs
                        strategy[pair_hand] = [0.2, 0.2, 0.3, 0.3, 0.0]  # Conservative

        # Ace hands
        for suit_ace in range(3):
            ace_card = 8 + (suit_ace * 9)
            for rank in range(8):  # 2-9 (excluding Ace)
                for suit in range(3):
                    if suit_ace == suit:  # Suited Ace
                        if rank >= 6:  # Ace with high card suited
                            hand = tuple(sorted([ace_card, rank + suit * 9]))
                            strategy[hand] = [0.1, 0.4, 0.1, 0.4, 0.0]  # Strong
                        elif rank >= 4:  # Ace with medium card suited
                            hand = tuple(sorted([ace_card, rank + suit * 9]))
                            strategy[hand] = [0.2, 0.3, 0.2, 0.3, 0.0]  # Moderate
                        else:  # Ace with low card suited
                            hand = tuple(sorted([ace_card, rank + suit * 9]))
                            strategy[hand] = [0.2, 0.2, 0.3, 0.3, 0.0]  # Conservative
                    else:  # Offsuit Ace
                        if rank >= 6:  # Ace with high card offsuit
                            hand = tuple(sorted([ace_card, rank + suit * 9]))
                            strategy[hand] = [0.2, 0.3, 0.2, 0.3, 0.0]  # Moderate
                        else:  # Ace with low card offsuit
                            hand = tuple(sorted([ace_card, rank + suit * 9]))
                            strategy[hand] = [0.3, 0.1, 0.4, 0.2, 0.0]  # Weak

        # High cards (8-9)
        for rank1 in range(6, 8):  # 8-9
            for suit1 in range(3):
                for rank2 in range(6, 8):  # 8-9
                    for suit2 in range(3):
                        if (
                            rank1 != rank2 and suit1 != suit2
                        ):  # Different ranks, offsuit
                            hand = tuple(sorted([rank1 + suit1 * 9, rank2 + suit2 * 9]))
                            strategy[hand] = [0.2, 0.2, 0.3, 0.3, 0.0]  # Moderate
                        elif (
                            rank1 != rank2 and suit1 == suit2
                        ):  # Different ranks, suited
                            hand = tuple(sorted([rank1 + suit1 * 9, rank2 + suit2 * 9]))
                            strategy[hand] = [0.1, 0.3, 0.2, 0.4, 0.0]  # Strong suited

        # Default strategy for unrecognized hands
        strategy["default"] = [0.3, 0.1, 0.3, 0.3, 0.0]  # Conservative default

        return strategy

    def _initialize_discard_ev_table(self):
        """Initialize expected value table for discard decisions."""
        ev_table = {}

        # For each possible hand combination
        for i in range(27):
            for j in range(i + 1, 27):
                hole_cards = tuple(sorted([i, j]))
                rank_i, rank_j = i % 9, j % 9

                if rank_i == rank_j:  # Pairs - don't discard
                    ev_table[hole_cards] = (1.0, -0.5, -0.5)
                elif rank_i == 8 or rank_j == 8:  # Ace + something - keep ace
                    if rank_i == 8:
                        ev_table[hole_cards] = (
                            0.3,
                            -0.5,
                            0.5,
                        )  # Keep Ace, discard other
                    else:
                        ev_table[hole_cards] = (
                            0.3,
                            0.5,
                            -0.5,
                        )  # Keep Ace, discard other
                elif min(rank_i, rank_j) < 4:  # One low card - discard it
                    if rank_i < rank_j:
                        ev_table[hole_cards] = (0.0, 0.5, -0.3)  # Discard low card
                    else:
                        ev_table[hole_cards] = (0.0, -0.3, 0.5)  # Discard low card
                else:  # Two decent cards - keep them
                    ev_table[hole_cards] = (0.3, -0.1, -0.1)

        # Default conservative strategy - only discard if significant improvement expected
        ev_table["default"] = (0.0, -0.2, -0.2)  # Slight preference for keeping cards

        return ev_table

    def _load_strategy_files(self):
        """Load pre-computed strategies and tuned parameters."""
        try:
            # Load strategy file
            if os.path.exists("data/precomputed_strategy.pkl"):
                with open("data/precomputed_strategy.pkl", "rb") as f:
                    precomputed = pickle.load(f)
                    self.regret_sum.update(precomputed.get("regret_sum", {}))
                    self.strategy_sum.update(precomputed.get("strategy_sum", {}))
                    self.iterations = precomputed.get("iterations", 0)

                    # Only use precomputed strategies if they seem valid
                    if self._verify_strategy_quality(precomputed):
                        if (
                            "preflop_strategy" in precomputed
                            and precomputed["preflop_strategy"]
                        ):
                            self.preflop_strategy.update(
                                precomputed["preflop_strategy"]
                            )

                        if (
                            "discard_ev_table" in precomputed
                            and precomputed["discard_ev_table"]
                        ):
                            self.discard_ev_table.update(
                                precomputed["discard_ev_table"]
                            )

                        self.logger.info(
                            f"Loaded pre-computed strategy with {self.iterations} iterations"
                        )
                    else:
                        self.logger.warning(
                            "Loaded strategy appears suboptimal, using baseline strategies"
                        )
            else:
                self.logger.info(
                    "No pre-computed strategy found, using baseline strategies"
                )

            # Load tuned parameters
            if os.path.exists("data/tuned_parameters.pkl"):
                with open("data/tuned_parameters.pkl", "rb") as f:
                    parameters = pickle.load(f)

                    # Apply tuned parameters
                    self.exploitation_factor = parameters.get(
                        "exploitation_factor", 0.3
                    )
                    self.confidence_threshold = (
                        10  # Start exploiting sooner with tuned parameters
                    )
                    self.discard_threshold = parameters.get("discard_threshold", 0.0)
                    self.aggression_factor = parameters.get("aggression_factor", 1.0)

                    self.logger.info(f"Loaded tuned parameters: {parameters}")

        except Exception as e:
            self.logger.error(f"Error loading strategy files: {e}")

    def _verify_strategy_quality(self, precomputed):
        """Check if loaded strategy seems reasonable."""
        # Check for empty or missing strategy components
        if (
            not precomputed.get("preflop_strategy")
            or len(precomputed.get("preflop_strategy", {})) < 5
        ):
            return False

        # Check if all strategies are identical (a sign of poor training)
        strategies = list(precomputed.get("preflop_strategy", {}).values())
        if len(strategies) > 5:
            # Compare a few random strategies to see if they're all identical
            first_strat = strategies[0]
            all_identical = True

            for strat in strategies[1:5]:
                if not isinstance(strat, list) or len(strat) != 5:
                    return False  # Invalid format

                if not np.allclose(first_strat, strat, atol=0.01):
                    all_identical = False
                    break

            if all_identical:
                return False  # All strategies are identical, likely poor training

        # Check discard EV table quality
        discard_table = precomputed.get("discard_ev_table", {})
        if len(discard_table) < 5:
            return False

        # Check a sample of discard EVs for pairs - should prefer keeping
        pair_found = False
        for key, values in discard_table.items():
            if key != "default" and isinstance(key, tuple) and len(key) == 2:
                if key[0] % 9 == key[1] % 9:  # It's a pair
                    pair_found = True
                    # For pairs, should prefer keeping over discarding
                    if not (values[0] > values[1] and values[0] > values[2]):
                        return False

        return pair_found  # At least one valid pair strategy found

    def get_strategy(self, infoset):
        """Get current strategy for an infoset based on accumulated regrets."""
        regrets = self.regret_sum[infoset]

        # Use regret-matching to compute strategy
        positive_regrets = np.maximum(regrets, 0)
        sum_positive_regrets = np.sum(positive_regrets)

        if sum_positive_regrets > 0:
            strategy = positive_regrets / sum_positive_regrets
        else:
            # If all regrets are negative or zero, use a uniform strategy
            strategy = np.ones(5) / 5

        return strategy

    def get_average_strategy(self, infoset):
        """Get average strategy for an infoset across all iterations."""
        strategy_sum = self.strategy_sum[infoset]
        total = np.sum(strategy_sum)

        if total > 0:
            return strategy_sum / total
        else:
            # If no accumulated strategy, return uniform
            return np.ones(5) / 5

    def _create_infoset_key(self, observation):
        """Create a key for the current information set."""
        street = observation["street"]
        acting_agent = observation["acting_agent"]

        # Get hole cards and filter out -1 values
        my_cards = tuple(sorted([c for c in observation["my_cards"] if c != -1]))

        # Only include visible community cards (filter out -1 values)
        visible_community = tuple(
            [c for c in observation["community_cards"] if c != -1]
        )

        # Abstracted hand strength based on cards
        hand_strength = self._evaluate_hand_strength(my_cards, visible_community)

        # Betting context
        pot_size = observation["my_bet"] + observation["opp_bet"]
        bet_difference = abs(observation["my_bet"] - observation["opp_bet"])

        # Pot odds if facing a bet
        pot_odds = 0
        if observation["my_bet"] < observation["opp_bet"]:
            call_amount = observation["opp_bet"] - observation["my_bet"]
            pot_odds = pot_size / call_amount if call_amount > 0 else 0

        # Create an abstract infoset key
        infoset = f"{street}:{bet_difference}:{hand_strength}:{pot_odds:.1f}"

        # If we're later in the hand, include some history
        if len(self.street_history) > 0:
            history_str = "".join(
                str(a) for a in self.street_history[-2:]
            )  # Last 2 actions
            infoset = f"{infoset}:{history_str}"

        return infoset

    def _evaluate_hand_strength(self, hole_cards, community_cards):
        """Evaluate the strength of the current hand."""
        # Filter out placeholder -1 values
        community = list(community_cards)  # Ensure community_cards is a list

        # If we don't have community cards yet, estimate based on hole cards
        if not community:
            # Pre-flop hand strength estimation
            ranks = [c % 9 for c in hole_cards]
            suits = [c // 9 for c in hole_cards]

            # Check for pairs
            if ranks[0] == ranks[1]:
                if ranks[0] == 8:  # Pair of aces
                    return "very_high"
                elif ranks[0] >= 6:  # High pair (8-9)
                    return "high"
                elif ranks[0] >= 4:  # Medium pair (6-7)
                    return "medium_high"
                else:  # Low pair
                    return "medium"

            # Check for suited cards
            suited = suits[0] == suits[1] if len(suits) >= 2 else False

            # Ace-based hands
            if 8 in ranks:
                if suited:
                    if min(ranks) >= 6:  # Ace + high card suited
                        return "high"
                    elif min(ranks) >= 4:  # Ace + medium card suited
                        return "medium_high"
                    else:  # Ace + low card suited
                        return "medium"
                else:
                    if min(ranks) >= 6:  # Ace + high card
                        return "medium_high"
                    elif min(ranks) >= 4:  # Ace + medium card
                        return "medium"
                    else:  # Ace + low card
                        return "medium_low"

            # Non-ace hands
            if suited:
                if min(ranks) >= 6:  # Two high cards suited
                    return "medium_high"
                elif min(ranks) >= 4:  # Two medium cards suited
                    return "medium"
                else:  # Low cards suited
                    return "medium_low"
            else:
                if min(ranks) >= 6:  # Two high cards
                    return "medium"
                elif min(ranks) >= 4:  # Two medium cards
                    return "medium_low"
                else:  # Low cards
                    return "low"

        # With community cards, use the hand evaluator
        return self.hand_evaluator.evaluate_relative_strength(hole_cards, community)

    def _should_discard(self, observation):
        """Determine if we should discard a card and which one."""
        # Only consider discard if it's allowed
        if not observation["valid_actions"][action_types.DISCARD.value]:
            return False, -1

        hole_cards = observation["my_cards"]
        community_cards = [c for c in observation["community_cards"] if c != -1]

        # Get hole cards as a tuple for lookup
        hole_tuple = tuple(sorted(hole_cards))

        # Check our discard EV table
        if hole_tuple in self.discard_ev_table:
            evs = self.discard_ev_table[hole_tuple]
            keep_ev = evs[0]
            discard_evs = [evs[1], evs[2]]

            # Apply discard threshold parameter from tuning
            threshold = self.discard_threshold

            # If either discard option exceeds the keep EV plus threshold, discard
            if max(discard_evs) > keep_ev + threshold:
                best_discard_idx = 0 if discard_evs[0] > discard_evs[1] else 1
                return True, best_discard_idx

        # Advanced heuristic fallback
        ranks = [c % 9 for c in hole_cards]

        # Don't discard aces or pairs
        if 8 in ranks or ranks[0] == ranks[1]:
            return False, -1

        # Find the lowest card to potentially discard
        lowest_idx = 0 if ranks[0] < ranks[1] else 1
        lowest_rank = ranks[lowest_idx]

        # Consider discarding low cards (2-5)
        if lowest_rank < 4:
            # Check if we have community cards
            if community_cards:
                # Assess if our hand connects with the board
                board_ranks = [c % 9 for c in community_cards]

                # If our low card matches board, keep it (potential pair)
                if lowest_rank in board_ranks:
                    return False, -1

                # If our other card matches board, definitely discard the low one
                other_idx = 1 - lowest_idx
                if ranks[other_idx] in board_ranks:
                    return True, lowest_idx

            # No board connection, discard low card
            return True, lowest_idx

        # Default: don't discard
        return False, -1

    def _update_opponent_model(self, observation, action, is_opponent_action=False):
        """Update our model of the opponent based on observed actions."""
        if not is_opponent_action:
            return

        street = observation["street"]

        # Count action frequencies
        self.opp_action_counts[street][action[0]] += 1

        # Special tracking for aggression
        if action[0] == action_types.RAISE.value:
            self.opp_betting_frequency[street] = (
                self.opp_betting_frequency.get(street, 0) * 0.8 + 0.2
            )  # Exponential moving average with more weight on recent actions

        # Track fold frequency
        if action[0] == action_types.FOLD.value:
            self.opp_fold_frequency[street] = (
                self.opp_fold_frequency.get(street, 0) * 0.8 + 0.2
            )

        # Track discard frequency
        if action[0] == action_types.DISCARD.value:
            self.opp_discard_frequency = self.opp_discard_frequency * 0.8 + 0.2

    def _adjust_strategy_for_opponent(self, strategy, observation):
        """Adjust strategy based on opponent modeling and tuned parameters."""
        # Get exploitation factor based on observed hands
        exploitation = min(
            0.8,
            self.exploitation_factor
            * (self.total_observed_hands / max(1, self.confidence_threshold)),
        )

        # If we haven't observed enough hands, stick closer to base strategy
        if self.total_observed_hands < self.confidence_threshold:
            # Just apply aggression tuning
            adjusted_strategy = strategy.copy()

            # Apply aggression factor to raise probability
            adjusted_strategy[action_types.RAISE.value] *= self.aggression_factor

            # Renormalize
            total = sum(adjusted_strategy)
            if total > 0:
                return [s / total for s in adjusted_strategy]
            return strategy

        # With more hands observed, apply full opponent modeling
        street = observation["street"]
        adjusted_strategy = strategy.copy()

        # Get opponent tendencies for the current street
        opp_fold_freq = self.opp_fold_frequency.get(street, 0.3)
        opp_aggression = self.opp_betting_frequency.get(street, 0.3)

        # Get hand strength for context-specific adjustments
        hand_strength = self._evaluate_hand_strength(
            [c for c in observation["my_cards"] if c != -1],
            [c for c in observation["community_cards"] if c != -1],
        )

        # Adjust strategy based on opponent tendencies
        if opp_fold_freq > 0.4:  # Opponent folds too much
            # Increase raising by aggression factor
            adjusted_strategy[action_types.RAISE.value] = min(
                0.8,
                adjusted_strategy[action_types.RAISE.value]
                * (1.2 + self.aggression_factor / 2),
            )
            # Decrease folding
            adjusted_strategy[action_types.FOLD.value] *= 0.5
        elif opp_fold_freq < 0.2:  # Opponent rarely folds
            # More selective with raises, fold more with weak hands
            if hand_strength in ["low", "medium_low"]:
                adjusted_strategy[action_types.FOLD.value] *= 1.5
                adjusted_strategy[action_types.RAISE.value] *= 0.5

        if opp_aggression > 0.6:  # Very aggressive opponent
            if hand_strength in ["low", "medium_low"]:
                # Against aggressive opponents, fold more with weak hands
                adjusted_strategy[action_types.FOLD.value] = max(
                    0.5, adjusted_strategy[action_types.FOLD.value] * 1.5
                )
                adjusted_strategy[action_types.RAISE.value] *= 0.3  # Bluff less
            elif hand_strength in ["high", "very_high"]:
                # With strong hands against aggressive opponents
                if self.aggression_factor < 1.0:  # We're tuned for conservative play
                    adjusted_strategy[action_types.CALL.value] = max(
                        0.6, adjusted_strategy[action_types.CALL.value] * 1.5
                    )
                    adjusted_strategy[action_types.RAISE.value] *= 0.7  # Slow-play more
                else:  # We're tuned for aggressive play
                    adjusted_strategy[
                        action_types.RAISE.value
                    ] *= self.aggression_factor  # Re-raise more

        # Normalize the adjusted strategy
        total = sum(adjusted_strategy)
        if total > 0:
            adjusted_strategy = [p / total for p in adjusted_strategy]

        # Blend between the original strategy and the adjusted one
        blended_strategy = [
            (1 - exploitation) * s + exploitation * a
            for s, a in zip(strategy, adjusted_strategy)
        ]

        # Re-normalize
        total = sum(blended_strategy)
        return [p / total for p in blended_strategy] if total > 0 else strategy

    def _manage_time_budget(self, observation):
        """Manage time budget for the hand."""
        # Update time budget based on observed time usage
        if self.hand_number > 10 and observation.get("time_left", 0) > 0:
            # Calculate conservative time per hand limit
            remaining_hands = max(1, 1000 - self.hand_number)
            conservative_time_per_hand = observation["time_left"] / (
                remaining_hands * 1.1
            )  # 10% safety margin

            # Adjust max time per hand, but not too aggressively
            self.max_time_per_hand = min(
                0.8,  # Never use more than 0.8s per hand on average
                max(0.05, min(self.max_time_per_hand, conservative_time_per_hand)),
            )

        # Check if we're running out of time for this hand
        elapsed = time.time() - self.hand_start_time
        return elapsed < self.max_time_per_hand

    def act(self, observation, reward, terminated, truncated, info):
        """Determine the next action to take in the game."""
        # Start timing the action
        if (
            self.hand_start_time == 0
            or "hand_number" in info
            and info["hand_number"] != self.hand_number
        ):
            self.hand_start_time = time.time()
            self.hand_number = info.get("hand_number", self.hand_number + 1)
            self.street_history = []
            self.actions_this_hand = []
            self.discard_performed = False

        # Check if this is a new street
        current_street = observation["street"]
        if len(self.street_history) == 0 or current_street != self.street_history[-1]:
            self.street_history.append(current_street)

        # Get the list of valid actions
        valid_actions = observation["valid_actions"]
        valid_action_indices = [
            i for i, is_valid in enumerate(valid_actions) if is_valid
        ]

        # Default values
        action_type = (
            action_types.CHECK.value
            if valid_actions[action_types.CHECK.value]
            else action_types.FOLD.value
        )
        raise_amount = 0
        card_to_discard = -1

        # Special handling for discard decisions
        should_discard, card_idx = self._should_discard(observation)
        if should_discard and valid_actions[action_types.DISCARD.value]:
            action_type = action_types.DISCARD.value
            card_to_discard = card_idx
            self.discard_performed = True

            # Log discard decision
            if self.logger:
                self.logger.info(
                    f"Hand {self.hand_number}: Discarding card {card_idx} "
                    f"({int_card_to_str(observation['my_cards'][card_idx])})"
                )

            return action_type, raise_amount, card_to_discard

        # Special case for pre-flop with known hands
        if current_street == 0:
            hole_tuple = tuple(sorted(observation["my_cards"]))
            if hole_tuple in self.preflop_strategy:
                strategy = np.array(self.preflop_strategy[hole_tuple])
            else:
                strategy = np.array(
                    self.preflop_strategy.get("default", [0.2, 0.2, 0.3, 0.3, 0.0])
                )
        else:
            # Create infoset key for strategy lookup
            infoset = self._create_infoset_key(observation)

            # Get strategy from CFR or use default if not available
            if infoset in self.strategy_sum and np.sum(self.strategy_sum[infoset]) > 0:
                strategy = self.get_average_strategy(infoset)
            else:
                # Default strategies based on street and hand strength
                hand_strength = self._evaluate_hand_strength(
                    [c for c in observation["my_cards"] if c != -1],
                    [c for c in observation["community_cards"] if c != -1],
                )

                # More nuanced default strategies
                if hand_strength in ["very_high", "high"]:
                    strategy = np.array([0.0, 0.5, 0.1, 0.4, 0.0])
                elif hand_strength in ["medium_high", "medium"]:
                    strategy = np.array([0.1, 0.3, 0.3, 0.3, 0.0])
                elif hand_strength == "medium_low":
                    strategy = np.array([0.2, 0.2, 0.3, 0.3, 0.0])
                else:  # low
                    strategy = np.array([0.4, 0.1, 0.4, 0.1, 0.0])

        # Adjust strategy based on opponent model
        adjusted_strategy = self._adjust_strategy_for_opponent(strategy, observation)

        # Filter strategy to only include valid actions
        valid_strategy = np.zeros(5)
        for i in valid_action_indices:
            valid_strategy[i] = adjusted_strategy[i]

        # Re-normalize
        strategy_sum = np.sum(valid_strategy)
        if strategy_sum > 0:
            valid_strategy = valid_strategy / strategy_sum
        else:
            # Fall back to uniform strategy over valid actions
            for i in valid_action_indices:
                valid_strategy[i] = 1.0 / len(valid_action_indices)

        # Choose action based on strategy
        action_types_array = np.array(valid_action_indices)
        action_probs = np.array([valid_strategy[i] for i in valid_action_indices])

        # Check remaining time and simplify if needed
        if not self._manage_time_budget(observation):
            # If we're running out of time, take the most probable action immediately
            action_type = action_types_array[np.argmax(action_probs)]
        else:
            # Otherwise sample from the strategy distribution
            action_type = np.random.choice(action_types_array, p=action_probs)

        # Handle raise amount if needed
        if action_type == action_types.RAISE.value:
            # Determine appropriate raise size based on hand strength and position
            hand_strength = self._evaluate_hand_strength(
                [c for c in observation["my_cards"] if c != -1],
                [c for c in observation["community_cards"] if c != -1],
            )

            # More nuanced raise sizing
            min_raise = observation["min_raise"]
            max_raise = observation["max_raise"]

            # Adjust raise_factor based on hand strength and street
            if "very_high" in hand_strength:
                if current_street >= 2:  # Turn/River
                    raise_factor = (
                        0.9  # Very strong bet with strong hand on later streets
                    )
                else:
                    raise_factor = 0.7  # Strong but not max on early streets
            elif "high" in hand_strength:
                raise_factor = 0.6
            elif "medium_high" in hand_strength:
                raise_factor = 0.4
            elif "medium" in hand_strength:
                raise_factor = 0.3
            else:
                raise_factor = 0.2  # Small raise with weak hands (possible bluff)

            # Apply aggression factor
            raise_factor = min(1.0, raise_factor * self.aggression_factor)

            # Calculate raise amount
            raise_range = max_raise - min_raise
            raise_amount = min_raise + int(raise_range * raise_factor)

            # Make sure it's within bounds
            raise_amount = max(min_raise, min(max_raise, raise_amount))

            # Occasional large raises with very strong hands
            if (
                "very_high" in hand_strength
                and random.random() < 0.3 * self.aggression_factor
            ):
                raise_amount = max_raise

        # Record the action for history tracking
        self.actions_this_hand.append(action_type)

        return action_type, raise_amount, card_to_discard

    def observe(self, observation, reward, terminated, truncated, info):
        """Process observations, especially opponent actions."""
        # Get the opponent's last action
        opp_last_action_str = observation.get("opp_last_action", "None")

        # Only process if there's an actual action to observe
        if opp_last_action_str != "None":
            # Convert the action string to action type
            try:
                opp_action_type = getattr(action_types, opp_last_action_str).value
                # Create a mock action tuple for opponent modeling
                opp_action = (
                    opp_action_type,
                    0,
                    -1,
                )  # We don't know the exact raise amount
                self._update_opponent_model(
                    observation, opp_action, is_opponent_action=True
                )
            except (AttributeError, ValueError):
                pass  # Invalid action string

        # If the hand has terminated, update our models and prepare for the next hand
        if terminated:
            self.total_observed_hands += 1

            # Record time used for this hand
            end_time = time.time()
            if self.hand_start_time > 0:
                time_used = end_time - self.hand_start_time
                self.time_used_per_hand.append(time_used)

                # Log time usage periodically
                if self.hand_number % 100 == 0:
                    avg_time = sum(self.time_used_per_hand[-100:]) / min(
                        100, len(self.time_used_per_hand)
                    )
                    self.logger.info(
                        f"Hand {self.hand_number}: Avg time per hand: {avg_time:.4f}s, "
                        f"Time left: {observation.get('time_left', 'unknown')}"
                    )

            # Gradually increase exploitation as we gather more data
            if self.total_observed_hands > self.confidence_threshold:
                # Increase exploitation factor gradually, maxing out at 0.7
                self.exploitation_factor = min(
                    0.7,
                    0.3
                    + 0.4
                    * (self.total_observed_hands - self.confidence_threshold)
                    / 200,
                )

            # Reset for next hand
            self.hand_start_time = 0
            self.street_history = []
            self.actions_this_hand = []
            self.discard_performed = False


class HandEvaluator:
    """Hand evaluator for 27-card deck poker."""

    def __init__(self):
        self.RANK_VALUES = {
            0: 2,
            1: 3,
            2: 4,
            3: 5,
            4: 6,
            5: 7,
            6: 8,
            7: 9,
            8: 14,  # Ace is high (14)
        }

    def evaluate_relative_strength(self, hole_cards, community_cards):
        """Evaluate hand strength and return a category."""
        # Convert card indices to rank and suit
        # FIX: Convert both to lists before concatenation
        hole_list = list(hole_cards)
        community_list = list(community_cards)
        all_cards = hole_list + community_list

        ranks = [card % 9 for card in all_cards]
        suits = [card // 9 for card in all_cards]

        # Count ranks and suits
        rank_count = {}
        for r in ranks:
            rank_count[r] = rank_count.get(r, 0) + 1

        suit_count = {}
        for s in suits:
            suit_count[s] = suit_count.get(s, 0) + 1

        # Check for various hand types
        three_of_kind = any(count >= 3 for count in rank_count.values())
        pairs = sum(1 for count in rank_count.values() if count >= 2)
        has_full_house = three_of_kind and pairs >= 2
        has_flush = any(count >= 5 for count in suit_count.values())

        # Straight check
        sorted_ranks = sorted(set(ranks))

        # Check for Ace low straight (A,2,3,4,5)
        has_straight = False
        if 8 in sorted_ranks:  # If we have an Ace
            temp_ranks = sorted_ranks.copy()
            # Check for low straight
            if all(r in temp_ranks for r in [0, 1, 2, 3]):  # If we have 2,3,4,5
                has_straight = True

        # Normal straight check (5+ consecutive ranks)
        if not has_straight:
            consecutive = 1
            max_consecutive = 1

            for i in range(1, len(sorted_ranks)):
                if sorted_ranks[i] == sorted_ranks[i - 1] + 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                elif sorted_ranks[i] != sorted_ranks[i - 1]:  # Skip duplicates
                    consecutive = 1

            has_straight = max_consecutive >= 5

        # Check for straight flush
        has_straight_flush = False
        if has_straight and has_flush:
            # Check if the straight and flush share enough cards
            for s in range(3):  # 3 suits
                suited_ranks = [r for r, suit in zip(ranks, suits) if suit == s]
                if len(suited_ranks) >= 5:
                    # Try to find a straight within these suited cards
                    sorted_suited = sorted(set(suited_ranks))

                    # Check for straight
                    consecutive = 1
                    max_consecutive = 1

                    for i in range(1, len(sorted_suited)):
                        if sorted_suited[i] == sorted_suited[i - 1] + 1:
                            consecutive += 1
                            max_consecutive = max(max_consecutive, consecutive)
                        elif (
                            sorted_suited[i] != sorted_suited[i - 1]
                        ):  # Skip duplicates
                            consecutive = 1

                    if max_consecutive >= 5:
                        has_straight_flush = True
                        break

        # Determine hand strength category
        if has_straight_flush:
            return "very_high"
        elif has_full_house:
            return "very_high"
        elif has_flush:
            return "high"
        elif has_straight:
            return "high"
        elif three_of_kind:
            return "medium_high"
        elif pairs >= 2:
            return "medium"
        elif pairs == 1:
            pair_rank = next(r for r, count in rank_count.items() if count >= 2)
            if pair_rank == 8:  # Pair of Aces
                return "medium_high"
            elif pair_rank >= 6:  # High pair
                return "medium"
            else:
                return "medium_low"
        else:
            # High card
            max_rank = max(ranks)
            if max_rank == 8:  # Ace high
                return "medium_low"
            elif max_rank >= 6:  # 8-9 high
                return "low"
            else:
                return "very_low"
