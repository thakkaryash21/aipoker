import numpy as np
import pickle
import time
import os
from collections import defaultdict
import random
from tqdm import tqdm
import argparse

# Card utilities
RANKS = "23456789A"
SUITS = "dhs"  # diamonds, hearts, spades


def int_card_to_str(card_int):
    rank = RANKS[card_int % len(RANKS)]
    suit = SUITS[card_int // len(RANKS)]
    return rank + suit


class PokerHandEvaluator:
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


class EnhancedCFRTrainer:
    def __init__(self, load_existing=True):
        """Initialize the CFR trainer with enhanced features."""
        # Strategy data structures
        self.regret_sum = defaultdict(lambda: np.zeros(5))
        self.strategy_sum = defaultdict(lambda: np.zeros(5))
        self.iterations = 0

        # Card abstraction
        self.card_clusters = self._create_card_abstraction()

        # Pre-flop strategy for initialization
        self.preflop_strategy = {}

        # Discard EV table
        self.discard_ev_table = {}

        # Hand evaluator
        self.hand_evaluator = PokerHandEvaluator()

        # Try to load existing strategies
        if load_existing and os.path.exists("precomputed_strategy.pkl"):
            self._load_strategy()

    def _create_card_abstraction(self):
        """Create detailed card clusters for abstraction."""
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

    def _load_strategy(self):
        """Load pre-computed strategy from file."""
        try:
            with open("precomputed_strategy.pkl", "rb") as f:
                precomputed = pickle.load(f)
                self.regret_sum.update(precomputed.get("regret_sum", {}))
                self.strategy_sum.update(precomputed.get("strategy_sum", {}))
                self.iterations = precomputed.get("iterations", 0)
                self.preflop_strategy = precomputed.get("preflop_strategy", {})
                self.discard_ev_table = precomputed.get("discard_ev_table", {})
                print(f"Loaded pre-computed strategy with {self.iterations} iterations")
        except Exception as e:
            print(f"Error loading pre-computed strategy: {e}")

    def save_strategy(self):
        """Save the current strategy to file."""
        # Convert defaultdicts to regular dicts for serialization
        regret_dict = {k: v for k, v in self.regret_sum.items()}
        strategy_dict = {k: v for k, v in self.strategy_sum.items()}

        precomputed = {
            "regret_sum": regret_dict,
            "strategy_sum": strategy_dict,
            "iterations": self.iterations,
            "preflop_strategy": self.preflop_strategy,
            "discard_ev_table": self.discard_ev_table,
        }

        with open("precomputed_strategy.pkl", "wb") as f:
            pickle.dump(precomputed, f)
        print(f"Saved strategy after {self.iterations} iterations")

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

    def train_cfr(self, num_iterations=100000, save_interval=10000):
        """Run CFR training for specified number of iterations."""
        print(f"Starting enhanced CFR training for {num_iterations} iterations...")
        start_time = time.time()

        for i in tqdm(range(num_iterations)):
            self._cfr_iteration()
            self.iterations += 1

            # Save periodically to avoid losing progress
            if (i + 1) % save_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Completed {i+1} iterations in {elapsed:.2f}s ({(i+1)/elapsed:.2f} it/s)"
                )
                self.save_strategy()

        # Final save
        self.save_strategy()

        # Generate specialized strategies
        self._generate_preflop_strategy()
        self._generate_discard_ev_table()

        # Validate the quality of our trained strategies
        self._validate_training_quality()

        # Save again with specialized strategies
        self.save_strategy()

        elapsed = time.time() - start_time
        print(
            f"Training completed in {elapsed:.2f}s ({num_iterations/elapsed:.2f} it/s)"
        )

    def _cfr_iteration(self):
        """Run a single CFR iteration with sampling."""
        # Use Monte Carlo sampling with variance reduction
        for _ in range(3):  # Multiple samples per iteration improves convergence
            self._sample_game_trajectory()

    def _sample_game_trajectory(self):
        """Sample a complete game trajectory and update strategies."""
        # Deal cards for the game
        deck = list(range(27))
        random.shuffle(deck)

        p0_cards = deck[:2]
        p1_cards = deck[2:4]
        community = deck[4:9]

        # Different combinations of visible community cards by street
        community_by_street = [
            [],  # Pre-flop
            community[:3],  # Flop
            community[:4],  # Turn
            community,  # River
        ]

        # Initialize game state
        player_pots = [0, 0]
        street = 0
        acting_player = random.randint(0, 1)  # Random first player
        pot_size = 3  # Small + big blind

        # Track reach probabilities
        reach_probs = [1.0, 1.0]

        # Action history
        action_history = []

        # Simulate a complete hand
        while street < 4:
            visible_cards = community_by_street[street]

            # Get player's cards
            cards = p0_cards if acting_player == 0 else p1_cards

            # Create infoset for current state
            infoset = self._create_infoset(
                cards,
                visible_cards,
                pot_size,
                player_pots[0],
                player_pots[1],
                street,
                action_history,
            )

            # Calculate counterfactual reach probability
            counterfactual_reach = reach_probs[1 - acting_player]

            # Get current strategy for this infoset
            strategy = self.get_strategy(infoset)

            # Apply regularization to encourage diverse strategies
            strategy = self._regularize_strategy(strategy, infoset)

            # Get valid actions
            valid_actions = self._get_valid_actions(player_pots, street, action_history)

            # Filter strategy to valid actions
            valid_strategy = np.zeros(5)
            for a in valid_actions:
                valid_strategy[a] = strategy[a]

            # Normalize valid strategy
            valid_sum = np.sum(valid_strategy)
            if valid_sum > 0:
                valid_strategy = valid_strategy / valid_sum
            else:
                valid_strategy = np.zeros(5)
                for a in valid_actions:
                    valid_strategy[a] = 1.0 / len(valid_actions)

            # Choose an action based on the strategy
            action_probs = np.array([valid_strategy[a] for a in valid_actions])
            action = np.random.choice(valid_actions, p=action_probs)

            # Update reach probability
            reach_probs[acting_player] *= valid_strategy[action]

            # Compute action utilities
            action_utils = self._compute_action_utilities(
                action,
                cards,
                p0_cards,
                p1_cards,
                community,
                street,
                player_pots,
                pot_size,
                action_history,
            )

            # Compute regrets and update strategy
            self._update_regrets_and_strategy(
                infoset,
                valid_actions,
                valid_strategy,
                action_utils,
                counterfactual_reach,
                reach_probs[acting_player],
            )

            # Apply action effects
            terminal, new_pot, new_player_pots, new_street = self._apply_action(
                action, street, pot_size, player_pots, action_history
            )

            # Update game state
            pot_size = new_pot
            player_pots = new_player_pots
            action_history.append(action)

            if terminal:
                break

            if new_street > street:
                street = new_street
                acting_player = 0  # Small blind acts first on new street
            else:
                acting_player = 1 - acting_player  # Switch players

    def _create_infoset(
        self,
        hole_cards,
        community_cards,
        pot_size,
        p0_pot,
        p1_pot,
        street,
        action_history,
    ):
        """Create a key for the current information set."""
        # Card abstraction - cluster cards by strength
        hole_clusters = tuple(sorted([self.card_clusters[c] for c in hole_cards]))

        # Hand strength approximation
        hand_strength = self._evaluate_hand_strength(hole_cards, community_cards)

        # Pot-related context
        bet_difference = abs(p0_pot - p1_pot)
        pot_odds = pot_size / bet_difference if bet_difference > 0 else 0

        # Action history abstraction (last few actions)
        history_str = (
            "".join(str(a) for a in action_history[-3:]) if action_history else ""
        )

        # Create infoset key
        return f"{street}:{hand_strength}:{bet_difference}:{pot_odds:.1f}:{history_str}"

    def _evaluate_hand_strength(self, hole_cards, community_cards):
        """Evaluate the strength of the current hand with finer categories."""
        # If no community cards, evaluate based on hole cards
        if not community_cards:
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
            suited = suits[0] == suits[1]

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

        # With community cards, evaluate relative hand strength
        hand_value = self.hand_evaluator.evaluate(hole_cards, community_cards)

        # Map hand values to strength categories
        if hand_value >= 8000:  # Straight flush
            return "very_high"
        elif hand_value >= 7000:  # Full house
            return "very_high"
        elif hand_value >= 6000:  # Flush
            return "high"
        elif hand_value >= 5000:  # Straight
            return "high"
        elif hand_value >= 4000:  # Three of a kind
            return "medium_high"
        elif hand_value >= 3000:  # Two pair
            return "medium"
        elif hand_value >= 2000:  # Pair
            pair_rank = hand_value - 2000
            if pair_rank == 8:  # Pair of aces
                return "medium_high"
            elif pair_rank >= 6:  # High pair
                return "medium"
            else:
                return "medium_low"
        else:  # High card
            high_card = hand_value - 1000
            if high_card == 8:  # Ace high
                return "medium_low"
            elif high_card >= 6:  # High card
                return "low"
            else:
                return "very_low"

    def _regularize_strategy(self, strategy, infoset):
        """Apply regularization to encourage diverse strategies."""
        # Add a small amount of exploration to every action
        epsilon = 0.05
        regularized = (1 - epsilon) * strategy + epsilon / 5

        # Ensure actions have at least some minimum probability
        min_prob = 0.02
        low_probs = regularized < min_prob

        if np.any(low_probs):
            # Increase probabilities below threshold
            deficit = np.sum(min_prob - regularized[low_probs])
            regularized[low_probs] = min_prob

            # Decrease other probabilities proportionally
            high_indices = ~low_probs
            if np.any(high_indices):
                total_high = np.sum(regularized[high_indices])
                if total_high > 0:
                    regularized[high_indices] *= (total_high - deficit) / total_high

        # Normalize
        total = np.sum(regularized)
        if total > 0:
            return regularized / total

        return strategy

    def _get_valid_actions(self, player_pots, street, action_history):
        """Get valid actions in current state."""
        valid_actions = []

        # Fold is always valid
        valid_actions.append(0)  # FOLD

        # Raise is valid unless at maximum bet
        if max(player_pots) < 100:
            valid_actions.append(1)  # RAISE

        # Check is valid if bets are equal
        if player_pots[0] == player_pots[1]:
            valid_actions.append(2)  # CHECK

        # Call is valid if bets are not equal
        if player_pots[0] != player_pots[1]:
            valid_actions.append(3)  # CALL

        # Discard is valid only in early streets and if not already used
        if street <= 1 and 4 not in action_history:
            valid_actions.append(4)  # DISCARD

        return valid_actions

    def _compute_action_utilities(
        self,
        action,
        cards,
        p0_cards,
        p1_cards,
        community,
        street,
        player_pots,
        pot_size,
        action_history,
    ):
        """Compute utilities for different actions."""
        # Set up utilities for all actions
        utilities = np.zeros(5) - 1000  # Default to very negative for invalid actions

        # For terminal states (fold or showdown), evaluate based on pot and hand strength
        if action == 0:  # FOLD
            utilities[0] = -min(player_pots)  # Lose the smaller bet
        elif street == 3:  # River - showdown
            # Compare hand strengths
            p0_value = self.hand_evaluator.evaluate(p0_cards, community)
            p1_value = self.hand_evaluator.evaluate(p1_cards, community)

            if p0_value > p1_value:
                utilities[1:4] = pot_size if cards == p0_cards else -pot_size
            elif p1_value > p0_value:
                utilities[1:4] = pot_size if cards == p1_cards else -pot_size
            else:  # Tie
                utilities[1:4] = 0
        else:
            # Non-terminal states - estimate EV based on hand strength
            hand_strength = self._evaluate_hand_strength(
                cards, community[: street + 2] if street > 0 else []
            )

            # Convert strength category to numeric value
            strength_values = {
                "very_high": 1.0,
                "high": 0.8,
                "medium_high": 0.6,
                "medium": 0.5,
                "medium_low": 0.3,
                "low": 0.2,
                "very_low": 0.1,
            }

            strength_value = strength_values.get(hand_strength, 0.5)

            # Calculate expected value based on strength and action
            if action == 1:  # RAISE
                utilities[1] = pot_size * (
                    strength_value * 2 - 0.5
                )  # Higher EV for strong hands
            elif action == 2:  # CHECK
                utilities[2] = pot_size * (
                    strength_value - 0.5
                )  # Lower EV than raising
            elif action == 3:  # CALL
                utilities[3] = pot_size * (
                    strength_value * 1.5 - 0.5
                )  # Between raise and check
            elif action == 4:  # DISCARD
                # Discard utility depends on current hand strength
                if hand_strength in ["low", "very_low", "medium_low"]:
                    utilities[4] = (
                        pot_size * 0.4
                    )  # High utility for discarding weak hands
                elif hand_strength in ["medium"]:
                    utilities[4] = pot_size * 0.1  # Neutral for medium hands
                else:
                    utilities[4] = (
                        -pot_size * 0.2
                    )  # Negative utility for discarding good hands

        return utilities

    def _update_regrets_and_strategy(
        self,
        infoset,
        valid_actions,
        strategy,
        utilities,
        counterfactual_reach,
        strategy_weight,
    ):
        """Update regrets and strategy for the infoset."""
        # Calculate expected value under current strategy
        ev = np.sum(strategy * utilities)

        # Update regrets for each action
        for action in valid_actions:
            regret = counterfactual_reach * (utilities[action] - ev)
            self.regret_sum[infoset][action] += regret

        # Add current strategy to accumulated strategy sum
        self.strategy_sum[infoset] += strategy_weight * strategy

    def _apply_action(self, action, street, pot_size, player_pots, action_history):
        """Apply action and return updated game state."""
        terminal = False
        new_player_pots = player_pots.copy()
        new_street = street

        if action == 0:  # FOLD
            terminal = True
        elif action == 1:  # RAISE
            # Simple raise logic - increase bet by 25% of pot
            bet_diff = abs(player_pots[0] - player_pots[1])
            raise_amount = max(2, int(pot_size * 0.25))

            # Apply raise to acting player
            acting_player = len(action_history) % 2  # Simple alternating players
            new_player_pots[acting_player] = max(player_pots) + raise_amount
        elif action == 2:  # CHECK
            # If both players check, advance to next street
            if action_history and action_history[-1] == 2:
                new_street = min(3, street + 1)
        elif action == 3:  # CALL
            # Match the opponent's bet
            new_player_pots[0] = new_player_pots[1] = max(player_pots)

            # After call, advance to next street
            new_street = min(3, street + 1)
        elif action == 4:  # DISCARD
            # Discard doesn't change bets or street
            pass

        # Calculate new pot size
        new_pot = sum(new_player_pots)

        return terminal, new_pot, new_player_pots, new_street

    def _generate_preflop_strategy(self):
        """Generate optimized pre-flop strategy table."""
        print("Generating pre-flop strategy table...")

        # Initialize stats to track strategy diversity
        strategy_count = 0
        unique_strategies = set()

        # Create all possible hole card combinations for preflop
        for i in range(27):
            for j in range(i + 1, 27):
                hole_cards = (i, j)

                # Create a sample infoset for preflop with these cards
                hole_clusters = tuple(
                    sorted([self.card_clusters[i], self.card_clusters[j]])
                )
                hand_strength = self._evaluate_hand_strength(hole_cards, [])

                # Find matching infosets
                matching_infosets = []
                for infoset in self.strategy_sum.keys():
                    if infoset.startswith(f"0:{hand_strength}:"):
                        matching_infosets.append(infoset)

                if matching_infosets:
                    # Average the strategies for these infosets
                    strategies = [
                        self.get_average_strategy(infoset)
                        for infoset in matching_infosets
                    ]
                    avg_strategy = np.mean(strategies, axis=0)

                    # Regularize for more diversity
                    avg_strategy = self._regularize_strategy(avg_strategy, None)

                    # Store in preflop strategy table
                    self.preflop_strategy[hole_cards] = avg_strategy.tolist()

                    # Track strategy diversity
                    strategy_count += 1
                    strategy_string = ",".join([f"{v:.2f}" for v in avg_strategy])
                    unique_strategies.add(strategy_string)
                else:
                    # Create default strategy based on hand strength
                    if hand_strength == "very_high":
                        self.preflop_strategy[hole_cards] = [0.0, 0.7, 0.0, 0.3, 0.0]
                    elif hand_strength == "high":
                        self.preflop_strategy[hole_cards] = [0.0, 0.5, 0.1, 0.4, 0.0]
                    elif hand_strength == "medium_high":
                        self.preflop_strategy[hole_cards] = [0.1, 0.4, 0.2, 0.3, 0.0]
                    elif hand_strength == "medium":
                        self.preflop_strategy[hole_cards] = [0.1, 0.3, 0.3, 0.3, 0.0]
                    elif hand_strength == "medium_low":
                        self.preflop_strategy[hole_cards] = [0.2, 0.2, 0.4, 0.2, 0.0]
                    else:  # low or very_low
                        self.preflop_strategy[hole_cards] = [0.3, 0.1, 0.5, 0.1, 0.0]

                    # Track diversity
                    strategy_count += 1

        # Add default strategy
        self.preflop_strategy["default"] = [0.2, 0.2, 0.3, 0.3, 0.0]

        print(f"Generated pre-flop strategies for {strategy_count} hand combinations")
        print(f"Strategy diversity: {len(unique_strategies)} unique strategies")

    def _generate_discard_ev_table(self):
        """Generate EV table for discard decisions with Monte Carlo simulation."""
        print("Generating discard EV table...")

        # Track discard EV diversity
        ev_count = 0
        unique_evs = set()

        # For each possible hole card combination
        for i in range(27):
            for j in range(i + 1, 27):
                hole_cards = (i, j)

                # Run a small Monte Carlo simulation for each discard option
                keep_ev = self._simulate_discard_ev(hole_cards, -1)
                discard_0_ev = self._simulate_discard_ev(hole_cards, 0)
                discard_1_ev = self._simulate_discard_ev(hole_cards, 1)

                # Create discard EV entry
                evs = (keep_ev, discard_0_ev, discard_1_ev)
                self.discard_ev_table[hole_cards] = evs

                # Track diversity
                ev_count += 1
                ev_string = ",".join([f"{v:.2f}" for v in evs])
                unique_evs.add(ev_string)

        # Add default strategy
        self.discard_ev_table["default"] = (0.0, -0.2, -0.2)

        print(f"Generated discard EV table for {ev_count} hand combinations")
        print(f"EV diversity: {len(unique_evs)} unique EV patterns")

    def _simulate_discard_ev(self, hole_cards, discard_idx):
        """Simulate expected value for a discard option."""
        # Number of simulations
        num_sims = 50
        total_ev = 0.0

        for _ in range(num_sims):
            # Create a deck without the hole cards
            deck = [c for c in range(27) if c not in hole_cards]
            random.shuffle(deck)

            # If discarding, replace the card
            sim_hole_cards = list(hole_cards)
            if discard_idx >= 0:
                sim_hole_cards[discard_idx] = deck.pop(0)

            # Deal community cards
            community = deck[:5]

            # Evaluate hand strength
            hand_value = self.hand_evaluator.evaluate(sim_hole_cards, community)

            # Deal opponent hand
            opponent_cards = deck[5:7]
            opponent_value = self.hand_evaluator.evaluate(opponent_cards, community)

            # Determine win/loss
            if hand_value > opponent_value:
                ev = 1.0  # Win
            elif hand_value < opponent_value:
                ev = -1.0  # Loss
            else:
                ev = 0.0  # Tie

            total_ev += ev

        # Return average EV
        return total_ev / num_sims

    def _validate_training_quality(self):
        """Validate the quality of trained strategies."""
        print("\nValidating training quality...")

        # Analyze preflop strategies
        if self.preflop_strategy:
            strats = np.array(list(self.preflop_strategy.values()))
            if "default" in self.preflop_strategy:
                strats = np.array(
                    [v for k, v in self.preflop_strategy.items() if k != "default"]
                )

            if len(strats) > 0:
                # Measure strategy diversity
                mean_strat = np.mean(strats, axis=0)
                std_strat = np.std(strats, axis=0)
                print(f"Preflop strategy statistics:")
                print(f"  Average: {mean_strat}")
                print(f"  Std Dev: {std_strat}")

                # Check for balanced actions
                print(
                    f"  Action balance: {np.sum(std_strat):.4f} (higher is more diverse)"
                )

                # Check for identical strategies
                unique_strats = set()
                for strat in strats:
                    strat_str = ",".join([f"{v:.2f}" for v in strat])
                    unique_strats.add(strat_str)

                print(f"  Unique strategies: {len(unique_strats)} / {len(strats)}")

                # Warn if too many identical strategies
                if len(unique_strats) < len(strats) * 0.5:
                    print("  WARNING: Low strategy diversity detected")

        # Analyze discard EV table
        if self.discard_ev_table:
            evs = np.array(list(self.discard_ev_table.values()))
            if "default" in self.discard_ev_table:
                evs = np.array(
                    [v for k, v in self.discard_ev_table.items() if k != "default"]
                )

            if len(evs) > 0:
                mean_ev = np.mean(evs, axis=0)
                std_ev = np.std(evs, axis=0)
                print(f"Discard EV statistics:")
                print(f"  Average: {mean_ev}")
                print(f"  Std Dev: {std_ev}")

                # Check for diversity
                unique_evs = set()
                for ev in evs:
                    ev_str = ",".join([f"{v:.2f}" for v in ev])
                    unique_evs.add(ev_str)

                print(f"  Unique EV patterns: {len(unique_evs)} / {len(evs)}")

                # Warn if too many identical EVs
                if len(unique_evs) < len(evs) * 0.5:
                    print("  WARNING: Low EV diversity detected")

        # Overall quality assessment
        quality_score = self._compute_quality_score()
        print(f"Overall training quality score: {quality_score:.2f}/10")

        if quality_score < 5:
            print("WARNING: Training quality is below acceptable threshold")
            print("Consider increasing iterations or adjusting parameters")

    def _compute_quality_score(self):
        """Compute an overall quality score for the training."""
        score = 0.0

        # Strategy diversity score (0-4)
        if self.preflop_strategy:
            strats = np.array(list(self.preflop_strategy.values()))
            if "default" in self.preflop_strategy:
                strats = np.array(
                    [v for k, v in self.preflop_strategy.items() if k != "default"]
                )

            if len(strats) > 0:
                # Count unique strategies
                unique_strats = set()
                for strat in strats:
                    strat_str = ",".join([f"{v:.2f}" for v in strat])
                    unique_strats.add(strat_str)

                diversity_ratio = len(unique_strats) / max(1, len(strats))
                score += diversity_ratio * 4

        # Discard EV quality score (0-3)
        if self.discard_ev_table:
            # Check that pair EVs favor keeping
            pair_count = 0
            correct_pairs = 0

            for cards, evs in self.discard_ev_table.items():
                if cards == "default":
                    continue

                if isinstance(cards, tuple) and len(cards) == 2:
                    if cards[0] % 9 == cards[1] % 9:  # It's a pair
                        pair_count += 1
                        if evs[0] > max(evs[1], evs[2]):  # Keep EV > discard EVs
                            correct_pairs += 1

            if pair_count > 0:
                pair_ratio = correct_pairs / pair_count
                score += pair_ratio * 3

        # Strategy balance score (0-3)
        num_infosets = len(self.strategy_sum)
        if num_infosets > 0:
            balanced_count = 0

            for infoset, strategy_sum in self.strategy_sum.items():
                if np.sum(strategy_sum) > 0:
                    avg_strategy = strategy_sum / np.sum(strategy_sum)
                    # Consider balanced if no action has probability > 0.7
                    if np.max(avg_strategy) < 0.7:
                        balanced_count += 1

            balance_ratio = balanced_count / num_infosets
            score += balance_ratio * 3

        return score


def train_cfr_bot(iterations=200000, save_interval=10000, load_existing=True):
    """Train the CFR bot with improved approach."""
    trainer = EnhancedCFRTrainer(load_existing=load_existing)
    trainer.train_cfr(num_iterations=iterations, save_interval=save_interval)
    return trainer


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train enhanced CFR poker bot")
    parser.add_argument(
        "--iterations", type=int, default=200000, help="Number of training iterations"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10000, help="Save interval"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh training (ignore existing strategy)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate existing strategy without training",
    )
    args = parser.parse_args()

    # Create trainer
    trainer = EnhancedCFRTrainer(load_existing=not args.fresh)

    if args.validate:
        # Only validate existing strategy
        trainer._validate_training_quality()
    else:
        # Run training
        trainer.train_cfr(
            num_iterations=args.iterations, save_interval=args.save_interval
        )

    # Display some sample strategies
    print("\nSample pre-flop strategies:")
    for hole_cards, strategy in random.sample(
        list(trainer.preflop_strategy.items()), min(5, len(trainer.preflop_strategy))
    ):
        if hole_cards != "default":
            card_strs = [int_card_to_str(card) for card in hole_cards]
            print(f"Hand {card_strs}: {strategy}")

    print("\nSample discard EVs:")
    for hole_cards, evs in random.sample(
        list(trainer.discard_ev_table.items()), min(5, len(trainer.discard_ev_table))
    ):
        if hole_cards != "default":
            card_strs = [int_card_to_str(card) for card in hole_cards]
            print(
                f"Hand {card_strs}: Keep EV={evs[0]:.2f}, Discard 1st={evs[1]:.2f}, Discard 2nd={evs[2]:.2f}"
            )
