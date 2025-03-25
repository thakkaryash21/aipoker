import numpy as np
import random
import pickle
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse


class PokerGameSimulator:
    """A poker game simulator for performance evaluation."""

    def __init__(self, strategy_file="precomputed_strategy.pkl"):
        self.load_strategy(strategy_file)
        self.hand_evaluator = HandEvaluator()

    def load_strategy(self, strategy_file):
        """Load a strategy from file."""
        if os.path.exists(strategy_file):
            with open(strategy_file, "rb") as f:
                self.strategy_data = pickle.load(f)
                print(
                    f"Loaded strategy with {self.strategy_data.get('iterations', 0)} iterations"
                )
        else:
            self.strategy_data = {
                "regret_sum": {},
                "strategy_sum": {},
                "iterations": 0,
                "preflop_strategy": {"default": [0.2, 0.2, 0.3, 0.3, 0.0]},
                "discard_ev_table": {"default": (0.0, -0.2, -0.2)},
            }
            print("No strategy file found, using defaults")

    def simulate_hand(self, strategy_params, opponent_type="random"):
        """
        Simulate a poker hand with specified strategy parameters.

        Args:
            strategy_params (dict): Parameters for adjusting the strategy
                - exploitation_factor: How much to adjust for opponent tendencies
                - discard_threshold: EV threshold for discarding cards
                - aggression_factor: Overall aggression level
                - betting_factor: Influences bet sizing
                - bluff_frequency: How often to bluff with weak hands

            opponent_type (str): Type of opponent to simulate against
                - "random": Makes random decisions
                - "tight": Plays conservatively, folds often
                - "aggressive": Raises frequently
                - "calling_station": Calls often, rarely folds

        Returns:
            float: Hand result (+1 for win, -1 for loss, 0 for tie)
        """
        # Create a simplified game environment
        deck = list(range(27))
        random.shuffle(deck)

        # Deal cards
        player_cards = [deck[0], deck[1]]
        opponent_cards = [deck[2], deck[3]]
        community = deck[4:9]

        # Track game state
        pot = 3  # Small + big blind
        player_bet = 1  # Small blind
        opponent_bet = 2  # Big blind
        street = 0
        player_discarded = False
        opponent_discarded = False

        # Simulate pre-flop
        result = self._simulate_street(
            player_cards,
            community[:0],
            pot,
            player_bet,
            opponent_bet,
            strategy_params,
            opponent_type,
            street=0,
        )
        if result is not None:
            return result

        # Update pot
        pot = player_bet + opponent_bet
        player_bet = opponent_bet = 0  # Reset bets for new street

        # Simulate discard decision
        should_discard, card_idx = self._evaluate_discard(
            player_cards, community[:3], strategy_params
        )
        if should_discard:
            player_cards[card_idx] = deck[9]  # Draw a new card
            player_discarded = True

        # Opponent discard decision (based on opponent type)
        opp_discard_probability = {
            "random": 0.3,
            "tight": 0.2,
            "aggressive": 0.3,
            "calling_station": 0.4,
        }.get(opponent_type, 0.3)

        if random.random() < opp_discard_probability:
            # Choose which card to discard based on opponent type
            opp_ranks = [c % 9 for c in opponent_cards]
            if opponent_type == "tight":
                # Only discard low cards
                if (
                    min(opp_ranks) < 4 and 8 not in opp_ranks
                ):  # Don't discard if have an Ace
                    opp_discard_idx = 0 if opp_ranks[0] < opp_ranks[1] else 1
                    opponent_cards[opp_discard_idx] = deck[10]
                    opponent_discarded = True
            else:
                # Simpler discard logic for other opponents
                opp_discard_idx = 0 if opp_ranks[0] < opp_ranks[1] else 1
                opponent_cards[opp_discard_idx] = deck[10]
                opponent_discarded = True

        # Simulate flop
        visible_community = community[:3]
        result = self._simulate_street(
            player_cards,
            visible_community,
            pot,
            player_bet,
            opponent_bet,
            strategy_params,
            opponent_type,
            street=1,
            player_discarded=player_discarded,
            opponent_discarded=opponent_discarded,
        )
        if result is not None:
            return result

        # Update pot for turn
        pot = player_bet + opponent_bet
        player_bet = opponent_bet = 0

        # Simulate turn
        visible_community = community[:4]
        result = self._simulate_street(
            player_cards,
            visible_community,
            pot,
            player_bet,
            opponent_bet,
            strategy_params,
            opponent_type,
            street=2,
        )
        if result is not None:
            return result

        # Update pot for river
        pot = player_bet + opponent_bet
        player_bet = opponent_bet = 0

        # Simulate river
        visible_community = community
        result = self._simulate_street(
            player_cards,
            visible_community,
            pot,
            player_bet,
            opponent_bet,
            strategy_params,
            opponent_type,
            street=3,
        )
        if result is not None:
            return result

        # If we reach here, there's a showdown
        player_hand_value = self.hand_evaluator.evaluate(player_cards, community)
        opponent_hand_value = self.hand_evaluator.evaluate(opponent_cards, community)

        if player_hand_value > opponent_hand_value:
            return pot  # Win pot
        elif opponent_hand_value > player_hand_value:
            return -pot  # Lose pot
        else:
            return 0  # Tie

    def _simulate_street(
        self,
        player_cards,
        visible_community,
        pot,
        player_bet,
        opponent_bet,
        strategy_params,
        opponent_type,
        street,
        player_discarded=False,
        opponent_discarded=False,
    ):
        """Simulate betting for one street."""
        # Generate opponent cards for simulation if needed
        # This was missing in the original code
        deck = list(range(27))
        for card in player_cards:
            if card in deck:
                deck.remove(card)
        for card in visible_community:
            if card in deck:
                deck.remove(card)
        random.shuffle(deck)
        opponent_cards = [deck[0], deck[1]]  # Assign opponent cards from remaining deck

        # Who acts first depends on the street
        if street == 0:  # Preflop - small blind (player) acts first
            first_actor = "player"
        else:  # Post-flop - big blind (opponent) acts first
            first_actor = "opponent"

        # First actor decision
        if first_actor == "player":
            result = self._player_decision(
                player_cards,
                visible_community,
                pot,
                player_bet,
                opponent_bet,
                strategy_params,
                street,
                player_discarded,
            )
            if isinstance(result, (int, float)):  # If terminal state
                return result
            player_bet, pot = result

            # Opponent responds
            result = self._opponent_decision(
                opponent_cards,
                visible_community,
                pot,
                player_bet,
                opponent_bet,
                opponent_type,
                street,
                opponent_discarded,
            )
            if isinstance(result, (int, float)):  # If terminal state
                return result
            opponent_bet, pot = result

            # If opponent raised, player gets to act again
            if opponent_bet > player_bet:
                result = self._player_decision(
                    player_cards,
                    visible_community,
                    pot,
                    player_bet,
                    opponent_bet,
                    strategy_params,
                    street,
                    player_discarded,
                )
                if isinstance(result, (int, float)):  # If terminal state
                    return result
                player_bet, pot = result
        else:
            # Opponent acts first
            result = self._opponent_decision(
                opponent_cards,
                visible_community,
                pot,
                player_bet,
                opponent_bet,
                opponent_type,
                street,
                opponent_discarded,
            )
            if isinstance(result, (int, float)):  # If terminal state
                return result
            opponent_bet, pot = result

            # Player responds
            result = self._player_decision(
                player_cards,
                visible_community,
                pot,
                player_bet,
                opponent_bet,
                strategy_params,
                street,
                player_discarded,
            )
            if isinstance(result, (int, float)):  # If terminal state
                return result
            player_bet, pot = result

            # If player raised, opponent gets to act again
            if player_bet > opponent_bet:
                result = self._opponent_decision(
                    opponent_cards,
                    visible_community,
                    pot,
                    player_bet,
                    opponent_bet,
                    opponent_type,
                    street,
                    opponent_discarded,
                )
                if isinstance(result, (int, float)):  # If terminal state
                    return result
                opponent_bet, pot = result

        # Update pot
        pot = player_bet + opponent_bet

        # Not terminal, continue to next street
        return None

    def _player_decision(
        self,
        player_cards,
        visible_community,
        pot,
        player_bet,
        opponent_bet,
        strategy_params,
        street,
        player_discarded,
    ):
        """Make a decision for the player based on strategy parameters."""
        # Create infoset for strategy lookup
        infoset = self._create_infoset(
            player_cards, visible_community, pot, player_bet, opponent_bet, street
        )

        # Check for pre-flop strategy
        if street == 0:
            hole_tuple = tuple(sorted(player_cards))
            if hole_tuple in self.strategy_data["preflop_strategy"]:
                strategy = np.array(self.strategy_data["preflop_strategy"][hole_tuple])
            else:
                strategy = np.array(
                    self.strategy_data["preflop_strategy"].get(
                        "default", [0.2, 0.2, 0.3, 0.3, 0.0]
                    )
                )
        else:
            # Get strategy from CFR or use default
            if infoset in self.strategy_data["strategy_sum"]:
                strategy_sum = self.strategy_data["strategy_sum"][infoset]
                total = np.sum(strategy_sum)
                if total > 0:
                    strategy = strategy_sum / total
                else:
                    strategy = np.array([0.2, 0.2, 0.3, 0.3, 0.0])  # Default
            else:
                # Default strategy based on hand strength
                strength = self._evaluate_hand_strength(player_cards, visible_community)

                if strength == "very_high":
                    strategy = np.array([0.0, 0.5, 0.1, 0.4, 0.0])
                elif strength == "high":
                    strategy = np.array([0.1, 0.4, 0.2, 0.3, 0.0])
                elif strength == "medium":
                    strategy = np.array([0.2, 0.2, 0.3, 0.3, 0.0])
                else:  # low
                    strategy = np.array([0.4, 0.1, 0.4, 0.1, 0.0])

        # Adjust strategy based on parameters
        adjusted_strategy = self._adjust_strategy(
            strategy,
            strategy_params,
            player_cards,
            visible_community,
            player_bet,
            opponent_bet,
        )

        # Get valid actions
        valid_actions = self._get_valid_actions(
            player_bet, opponent_bet, street, player_discarded
        )

        # Filter strategy to valid actions
        valid_strategy = np.zeros(5)
        for a in valid_actions:
            valid_strategy[a] = adjusted_strategy[a]

        # Normalize
        strategy_sum = np.sum(valid_strategy)
        if strategy_sum > 0:
            valid_strategy = valid_strategy / strategy_sum
        else:
            # Uniform over valid actions
            valid_strategy = np.zeros(5)
            for a in valid_actions:
                valid_strategy[a] = 1.0 / len(valid_actions)

        # Choose action based on strategy
        action_probs = [valid_strategy[a] for a in valid_actions]
        action = random.choices(valid_actions, weights=action_probs)[0]

        # Apply action
        if action == 0:  # FOLD
            return -min(player_bet, opponent_bet)  # Return -pot
        elif action == 1:  # RAISE
            # Determine raise size based on parameters
            min_raise = 2  # Minimum raise
            max_raise = 100 - opponent_bet  # Maximum possible raise

            # Calculate raise based on hand strength and aggression
            strength = self._evaluate_hand_strength(player_cards, visible_community)
            aggression = strategy_params.get("aggression_factor", 1.0)
            betting_factor = strategy_params.get("betting_factor", 1.0)

            # Map strength to a base raise factor
            strength_factor = {
                "very_high": 0.8,
                "high": 0.6,
                "medium": 0.4,
                "low": 0.2,
            }.get(strength, 0.5)

            # Adjust with parameters
            raise_factor = strength_factor * aggression * betting_factor

            # Calculate raise amount
            raise_amount = min_raise + int((max_raise - min_raise) * raise_factor)
            raise_amount = max(min_raise, min(max_raise, raise_amount))

            player_bet = opponent_bet + raise_amount
            pot = player_bet + opponent_bet

            return player_bet, pot
        elif action == 2:  # CHECK
            # Betting stays the same
            return player_bet, pot
        elif action == 3:  # CALL
            player_bet = opponent_bet
            pot = player_bet + opponent_bet
            return player_bet, pot
        else:  # DISCARD - doesn't change betting
            return player_bet, pot

    def _opponent_decision(
        self,
        opponent_cards,
        visible_community,
        pot,
        player_bet,
        opponent_bet,
        opponent_type,
        street,
        opponent_discarded,
    ):
        """Make a decision for the opponent based on type."""
        # Get valid actions
        valid_actions = self._get_valid_actions(
            opponent_bet, player_bet, street, opponent_discarded
        )

        # Evaluate opponent hand strength
        strength = self._evaluate_hand_strength(opponent_cards, visible_community)

        # Different behavior based on opponent type
        if opponent_type == "random":
            # Choose randomly from valid actions
            action = random.choice(valid_actions)
        elif opponent_type == "tight":
            # More likely to fold with weak hands
            if (
                strength in ["low", "medium"]
                and 0 in valid_actions
                and player_bet > opponent_bet
            ):
                action = random.choices([0, 3], weights=[0.7, 0.3])[
                    0
                ]  # 70% fold, 30% call
            elif strength == "high" and 1 in valid_actions:
                action = random.choices([1, 2, 3], weights=[0.3, 0.3, 0.4])[
                    0
                ]  # 30% raise
            elif strength == "very_high" and 1 in valid_actions:
                action = random.choices([1, 2, 3], weights=[0.6, 0.2, 0.2])[
                    0
                ]  # 60% raise
            else:
                action = random.choice(valid_actions)
        elif opponent_type == "aggressive":
            # More likely to raise with any hand
            if 1 in valid_actions:
                action = random.choices([0, 1, 2, 3], weights=[0.1, 0.6, 0.1, 0.2])[
                    0
                ]  # 60% raise
            else:
                action = random.choice(valid_actions)
        elif opponent_type == "calling_station":
            # Almost always calls, rarely folds
            if 3 in valid_actions:
                action = 3  # Always call if possible
            elif 0 in valid_actions and 2 in valid_actions:
                action = random.choices([0, 2], weights=[0.1, 0.9])[
                    0
                ]  # 90% check, 10% fold
            else:
                action = random.choice(valid_actions)
        else:
            # Default: balanced play
            if (
                strength in ["low", "medium"]
                and 0 in valid_actions
                and player_bet > opponent_bet
            ):
                action = random.choices([0, 3], weights=[0.5, 0.5])[
                    0
                ]  # 50% fold, 50% call
            elif strength == "high" and 1 in valid_actions:
                action = random.choices([1, 2, 3], weights=[0.4, 0.3, 0.3])[
                    0
                ]  # 40% raise
            elif strength == "very_high" and 1 in valid_actions:
                action = random.choices([1, 2, 3], weights=[0.7, 0.1, 0.2])[
                    0
                ]  # 70% raise
            else:
                action = random.choice(valid_actions)

        # Apply action
        if action == 0:  # FOLD
            return pot  # Return +pot (player wins)
        elif action == 1:  # RAISE
            # Determine raise size based on opponent type and hand
            min_raise = 2  # Minimum raise
            max_raise = 100 - player_bet  # Maximum possible raise

            # Set aggression level based on opponent type
            aggression = {
                "random": 0.5,
                "tight": 0.3,
                "aggressive": 0.8,
                "calling_station": 0.4,
            }.get(opponent_type, 0.5)

            # Map strength to a base raise factor
            strength_factor = {
                "very_high": 0.8,
                "high": 0.6,
                "medium": 0.4,
                "low": 0.2,
            }.get(strength, 0.5)

            # Calculate raise amount
            raise_factor = strength_factor * aggression
            raise_amount = min_raise + int((max_raise - min_raise) * raise_factor)
            raise_amount = max(min_raise, min(max_raise, raise_amount))

            opponent_bet = player_bet + raise_amount
            pot = player_bet + opponent_bet

            return opponent_bet, pot
        elif action == 2:  # CHECK
            # Betting stays the same
            return opponent_bet, pot
        elif action == 3:  # CALL
            opponent_bet = player_bet
            pot = player_bet + opponent_bet
            return opponent_bet, pot
        else:  # DISCARD - doesn't change betting
            return opponent_bet, pot

    def _create_infoset(
        self, hole_cards, community_cards, pot, player_bet, opponent_bet, street
    ):
        """Create infoset key for the current state."""
        # Hand strength abstraction
        hand_strength = self._evaluate_hand_strength(hole_cards, community_cards)

        # Betting pattern abstraction
        bet_diff = abs(player_bet - opponent_bet)
        pot_odds = pot / bet_diff if bet_diff > 0 else 0

        # Create infoset key
        return f"{street}:{hand_strength}:{bet_diff}:{pot_odds:.1f}"

    def _evaluate_hand_strength(self, hole_cards, community_cards):
        """Evaluate hand strength category."""
        # If no community cards, evaluate based on hole cards
        if not community_cards:
            ranks = [c % 9 for c in hole_cards]

            # Check for pairs
            if ranks[0] == ranks[1]:
                if ranks[0] == 8:  # Pair of aces
                    return "very_high"
                elif ranks[0] >= 6:  # High pair
                    return "high"
                else:  # Low/medium pair
                    return "medium"

            # Check for high cards
            if 8 in ranks:  # Ace
                return "medium"
            elif max(ranks) >= 6:  # High card
                return "medium"
            else:
                return "low"

        # With community cards, use hand evaluator
        hand_value = self.hand_evaluator.evaluate(hole_cards, community_cards)

        # Map to strength categories
        if hand_value >= 7000:  # Straight flush or full house
            return "very_high"
        elif hand_value >= 5000:  # Flush or straight
            return "high"
        elif hand_value >= 3000:  # Three of a kind or two pair
            return "medium"
        else:
            return "low"

    def _evaluate_discard(self, hole_cards, community_cards, strategy_params):
        """Decide whether to discard a card and which one."""
        # Get hole cards as a tuple for lookup
        hole_tuple = tuple(sorted(hole_cards))

        # Apply discard threshold parameter
        threshold = strategy_params.get("discard_threshold", 0.0)

        # Check EV table
        if hole_tuple in self.strategy_data["discard_ev_table"]:
            evs = self.strategy_data["discard_ev_table"][hole_tuple]
            keep_ev = evs[0]
            discard_evs = [evs[1], evs[2]]

            # If either discard option exceeds threshold, discard
            if max(discard_evs) > keep_ev + threshold:
                best_discard_idx = 0 if discard_evs[0] > discard_evs[1] else 1
                return True, best_discard_idx

        # Default fallback strategy
        ranks = [c % 9 for c in hole_cards]

        # Don't discard aces or pairs
        if 8 in ranks or ranks[0] == ranks[1]:
            return False, -1

        # Find the lowest card to potentially discard
        lowest_idx = 0 if ranks[0] < ranks[1] else 1
        lowest_rank = ranks[lowest_idx]

        # Discard threshold influences willingness to discard
        if lowest_rank < 4 - (threshold * 2):
            # Check if we have community cards
            if community_cards:
                # Check for potential pairs
                comm_ranks = [c % 9 for c in community_cards]
                if lowest_rank in comm_ranks:
                    return False, -1  # Don't discard if matching community
                if ranks[1 - lowest_idx] in comm_ranks:
                    return (
                        True,
                        lowest_idx,
                    )  # Definitely discard low card if other matches

            # No community matches, discard low card
            return True, lowest_idx

        # Default: don't discard
        return False, -1

    def _adjust_strategy(
        self, strategy, params, hole_cards, community_cards, player_bet, opponent_bet
    ):
        """Adjust strategy based on parameters."""
        adjusted = strategy.copy()

        # Extract parameters
        exploitation = params.get("exploitation_factor", 0.3)
        aggression = params.get("aggression_factor", 1.0)
        bluff_freq = params.get("bluff_frequency", 0.2)

        # Hand strength for context
        strength = self._evaluate_hand_strength(hole_cards, community_cards)

        # Adjust based on parameters
        if strength in ["very_high", "high"]:
            # With strong hands, adjust raise/call balance based on aggression
            if aggression > 1.0:
                # More aggressive: raise more
                adjusted[1] *= aggression  # RAISE
                adjusted[3] /= aggression  # CALL
            else:
                # More passive: call more, trap
                adjusted[1] *= aggression  # RAISE
                adjusted[3] *= 2 - aggression  # CALL
        elif strength == "medium":
            # Medium hands, balanced adjustment
            adjusted[1] *= aggression  # RAISE
        else:  # weak hands
            # Adjust bluffing frequency
            if random.random() < bluff_freq:
                # Occasional bluffs
                adjusted[1] *= aggression * bluff_freq * 5  # RAISE
            else:
                # Usually conservative with weak hands
                adjusted[0] *= 2 - aggression  # FOLD
                adjusted[1] *= 0.1  # RAISE

        # Normalize the adjusted strategy
        total = sum(adjusted)
        if total > 0:
            return adjusted / total
        return strategy

    def _get_valid_actions(self, actor_bet, opponent_bet, street, has_discarded):
        """Get valid actions for the current state."""
        valid_actions = []

        # Fold is valid if facing a bet
        if opponent_bet > actor_bet:
            valid_actions.append(0)  # FOLD

        # Raise is valid unless at max bet
        if actor_bet < 100 and opponent_bet < 100:
            valid_actions.append(1)  # RAISE

        # Check is valid if bets are equal
        if actor_bet == opponent_bet:
            valid_actions.append(2)  # CHECK

        # Call is valid if opponent has bet more
        if opponent_bet > actor_bet:
            valid_actions.append(3)  # CALL

        # Discard is valid in early streets if not already used
        if street <= 1 and not has_discarded:
            valid_actions.append(4)  # DISCARD

        return valid_actions


class HandEvaluator:
    """Evaluator for poker hand strength."""

    def evaluate(self, hole_cards, community_cards):
        """Evaluate the strength of a poker hand."""
        all_cards = hole_cards + community_cards
        ranks = [c % 9 for c in all_cards]
        suits = [c // 9 for c in all_cards]

        # Count ranks and suits
        rank_count = defaultdict(int)
        for r in ranks:
            rank_count[r] += 1

        suit_count = defaultdict(int)
        for s in suits:
            suit_count[s] += 1

        # Check for straight flush (best hand in 27-card deck)
        has_straight_flush = False
        for s in range(3):  # 3 suits
            suited_ranks = sorted(
                set([r for r, suit in zip(ranks, suits) if suit == s])
            )
            if len(suited_ranks) >= 5:
                # Check for straight
                has_straight_flush = self._check_straight(suited_ranks)
                if has_straight_flush:
                    return 8000 + max(suited_ranks)  # Straight flush

        # Check for full house
        three_of_a_kind = [r for r, count in rank_count.items() if count >= 3]
        pairs = [r for r, count in rank_count.items() if count >= 2]

        if three_of_a_kind and len(pairs) >= 2:
            return (
                7000
                + max(three_of_a_kind) * 100
                + max([p for p in pairs if p != max(three_of_a_kind)])
            )  # Full house

        # Check for flush
        for s, count in suit_count.items():
            if count >= 5:
                suited_ranks = [r for r, suit in zip(ranks, suits) if suit == s]
                return 6000 + sum(sorted(suited_ranks)[-5:])  # Flush

        # Check for straight
        distinct_ranks = sorted(set(ranks))
        if self._check_straight(distinct_ranks):
            return 5000 + max(self._get_straight_values(distinct_ranks))  # Straight

        # Check for three of a kind
        if three_of_a_kind:
            return 4000 + max(three_of_a_kind)  # Three of a kind

        # Check for two pair
        if len(pairs) >= 2:
            top_pairs = sorted(pairs)[-2:]
            return 3000 + top_pairs[1] * 100 + top_pairs[0]  # Two pair

        # Check for pair
        if pairs:
            return 2000 + max(pairs)  # Pair

        # High card
        return 1000 + max(ranks)  # High card

    def _check_straight(self, ranks):
        """Check if sorted ranks contain a straight."""
        if len(ranks) < 5:
            return False

        # Handle Ace (8) low straight (A,2,3,4,5)
        if 8 in ranks and 0 in ranks and 1 in ranks and 2 in ranks and 3 in ranks:
            return True

        # Normal straight check
        consecutive = 1
        max_consecutive = 1

        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i - 1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            elif ranks[i] > ranks[i - 1] + 1:  # Skip duplicates, reset on gap
                consecutive = 1

        return max_consecutive >= 5

    def _get_straight_values(self, ranks):
        """Get values in a straight."""
        # Check for A-5 straight
        if 8 in ranks and 0 in ranks and 1 in ranks and 2 in ranks and 3 in ranks:
            return [0, 1, 2, 3, 8]  # A-5 straight

        # Find longest consecutive sequence
        straights = []
        consecutive = [ranks[0]]

        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i - 1] + 1:
                consecutive.append(ranks[i])
            elif ranks[i] > ranks[i - 1] + 1:
                if len(consecutive) >= 5:
                    straights.append(consecutive[-5:])
                consecutive = [ranks[i]]

        if len(consecutive) >= 5:
            straights.append(consecutive[-5:])

        if straights:
            return max(straights, key=lambda x: x[-1])
        return []


def evaluate_parameters(params_with_id):
    """Evaluate a parameter set against different opponent types."""
    param_id, params = params_with_id
    simulator = PokerGameSimulator()

    # Number of hands to simulate per opponent type
    hands_per_type = 50

    # Different opponent types to test against
    opponent_types = ["random", "tight", "aggressive", "calling_station"]

    total_profit = 0

    # Run simulations against each opponent type
    for opp_type in opponent_types:
        profit = 0
        for _ in range(hands_per_type):
            result = simulator.simulate_hand(params, opponent_type=opp_type)
            profit += result

        # Track total profit
        total_profit += profit

    # Calculate average profit per hand
    avg_profit = total_profit / (len(opponent_types) * hands_per_type)

    return param_id, avg_profit, params


def tune_parameters(iterations=30, population_size=20, parallelism=4):
    """Use a genetic algorithm to tune strategy parameters."""
    print(
        f"Tuning parameters with {iterations} iterations, population size {population_size}"
    )

    # Parameter definitions and ranges
    param_specs = {
        "exploitation_factor": (0.0, 1.0),  # How much to adapt to opponent
        "discard_threshold": (-0.5, 0.5),  # Threshold for discard decisions
        "aggression_factor": (0.5, 2.0),  # Overall aggression level
        "betting_factor": (0.5, 2.0),  # Influences bet sizing
        "bluff_frequency": (0.0, 0.5),  # How often to bluff with weak hands
    }

    # Initialize population
    population = []
    for i in range(population_size):
        params = {}
        for param_name, (min_val, max_val) in param_specs.items():
            params[param_name] = min_val + random.random() * (max_val - min_val)
        population.append(params)

    best_score = -float("inf")
    best_params = None

    # Use process pool for parallel evaluation
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        for iteration in range(iterations):
            print(f"\nIteration {iteration+1}/{iterations}")

            # Evaluate all parameter sets in parallel
            param_sets_with_ids = [(i, params) for i, params in enumerate(population)]
            results = list(executor.map(evaluate_parameters, param_sets_with_ids))

            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)

            # Update best parameters if improved
            if results[0][1] > best_score:
                best_score = results[0][1]
                best_params = results[0][2].copy()
                print(f"New best score: {best_score:.4f}")
                print(f"Parameters: {best_params}")

            # Create next generation
            elite_size = max(1, population_size // 5)
            elites = [results[i][2].copy() for i in range(elite_size)]

            # Create children through crossover and mutation
            children = []

            # Add elites directly
            children.extend(elites)

            # Create remaining children
            while len(children) < population_size:
                # Tournament selection
                parent1 = random.choice(elites)
                parent2 = random.choice([r[2] for r in results])

                # Crossover
                child = {}
                for key in parent1:
                    if random.random() < 0.5:
                        child[key] = parent1[key]
                    else:
                        child[key] = parent2[key]

                # Mutation
                for key in child:
                    if random.random() < 0.3:  # 30% mutation rate
                        min_val, max_val = param_specs[key]
                        mutation_range = (max_val - min_val) * 0.2  # 20% of range
                        child[key] = max(
                            min_val,
                            min(
                                max_val,
                                child[key]
                                + random.uniform(-mutation_range, mutation_range),
                            ),
                        )

                children.append(child)

            # Update population
            population = children

            # Save best parameters after each iteration
            with open("tuned_parameters.pkl", "wb") as f:
                pickle.dump(best_params, f)

    print("\nParameter tuning complete")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Save final best parameters
    with open("tuned_parameters.pkl", "wb") as f:
        pickle.dump(best_params, f)
    print("Best parameters saved to tuned_parameters.pkl")

    return best_params


def test_parameters(params=None):
    """Test a set of parameters or the best saved parameters."""
    if params is None:
        # Load saved parameters
        if os.path.exists("tuned_parameters.pkl"):
            with open("tuned_parameters.pkl", "rb") as f:
                params = pickle.load(f)
            print(f"Testing saved parameters: {params}")
        else:
            # Default parameters
            params = {
                "exploitation_factor": 0.3,
                "discard_threshold": 0.0,
                "aggression_factor": 1.0,
                "betting_factor": 1.0,
                "bluff_frequency": 0.2,
            }
            print(f"Testing default parameters: {params}")

    # Run test simulations
    simulator = PokerGameSimulator()

    # Test against different opponents
    for opponent_type in ["random", "tight", "aggressive", "calling_station"]:
        total_profit = 0
        hands = 200

        for i in range(hands):
            result = simulator.simulate_hand(params, opponent_type=opponent_type)
            total_profit += result

        avg_profit = total_profit / hands
        print(
            f"Against {opponent_type} opponent: {avg_profit:.4f} avg profit ({total_profit} total)"
        )

    # Overall performance
    print(f"\nOverall performance summary for params: {params}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Tune and test poker strategy parameters"
    )
    parser.add_argument("--tune", action="store_true", help="Run parameter tuning")
    parser.add_argument("--test", action="store_true", help="Test parameters")
    parser.add_argument(
        "--iterations", type=int, default=30, help="Number of iterations for tuning"
    )
    parser.add_argument(
        "--population", type=int, default=20, help="Population size for tuning"
    )
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of parallel workers"
    )
    args = parser.parse_args()

    if args.tune:
        # Run parameter tuning
        best_params = tune_parameters(
            iterations=args.iterations,
            population_size=args.population,
            parallelism=args.parallel,
        )

        # Test the tuned parameters
        test_parameters(best_params)
    elif args.test:
        # Just test saved parameters
        test_parameters()
    else:
        print("Please specify either --tune or --test")
