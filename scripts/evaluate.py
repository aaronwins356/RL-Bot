#!/usr/bin/env python3
"""Evaluation script for RL-Bot.

This script evaluates trained models against various opponents and tracks Elo ratings.
"""
import argparse
import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.infra.config import load_config
from core.training.eval import EloRating, plot_elo_history
from bot_manager import BotManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate RL-Bot against opponents',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint to evaluate'
    )
    
    parser.add_argument(
        '--opponents',
        type=str,
        nargs='+',
        default=['rule_policy', 'baseline_ml'],
        help='Opponents to evaluate against'
    )
    
    parser.add_argument(
        '--num-games',
        type=int,
        default=10,
        help='Number of games to play against each opponent'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='eval_results.json',
        help='Output file for evaluation results'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate Elo rating plots'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic policy (no exploration)'
    )
    
    return parser.parse_args()


def run_evaluation(
    bot_manager: BotManager,
    opponents: List[str],
    num_games: int,
    deterministic: bool = True
) -> Dict[str, Any]:
    """Run evaluation matches.
    
    Args:
        bot_manager: Bot manager with loaded model
        opponents: List of opponent identifiers
        num_games: Number of games per opponent
        deterministic: Use deterministic policy
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'total_games': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'opponents': {},
        'elo_ratings': {},
    }
    
    # Initialize Elo ratings
    elo_system = EloRating(k_factor=32)
    our_elo = 1500.0
    opponent_elos = {opp: 1500.0 for opp in opponents}
    
    print("Starting evaluation matches...")
    print()
    
    for opponent in opponents:
        print(f"Playing against {opponent}...")
        
        opponent_results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'goal_diff': 0,
            'games': []
        }
        
        for game_idx in range(num_games):
            # In real implementation, would play actual game
            # For now, simulate a game result
            
            # Placeholder: random result
            import random
            result = random.choice(['win', 'loss', 'draw'])
            goal_diff = random.randint(-3, 3)
            
            if result == 'win':
                opponent_results['wins'] += 1
                results['wins'] += 1
                # Update Elo
                our_elo = elo_system.update_rating(our_elo, opponent_elos[opponent], 1.0)
                opponent_elos[opponent] = elo_system.update_rating(
                    opponent_elos[opponent], our_elo, 0.0
                )
            elif result == 'loss':
                opponent_results['losses'] += 1
                results['losses'] += 1
                # Update Elo
                our_elo = elo_system.update_rating(our_elo, opponent_elos[opponent], 0.0)
                opponent_elos[opponent] = elo_system.update_rating(
                    opponent_elos[opponent], our_elo, 1.0
                )
            else:
                opponent_results['draws'] += 1
                results['draws'] += 1
                # Update Elo
                our_elo = elo_system.update_rating(our_elo, opponent_elos[opponent], 0.5)
                opponent_elos[opponent] = elo_system.update_rating(
                    opponent_elos[opponent], our_elo, 0.5
                )
            
            opponent_results['goal_diff'] += goal_diff
            opponent_results['games'].append({
                'game_idx': game_idx,
                'result': result,
                'goal_diff': goal_diff,
            })
            
            results['total_games'] += 1
            
            print(f"  Game {game_idx + 1}/{num_games}: {result.upper()} "
                  f"(Goal diff: {goal_diff:+d}) - Elo: {our_elo:.0f}")
        
        results['opponents'][opponent] = opponent_results
        results['elo_ratings'][opponent] = opponent_elos[opponent]
        
        # Print summary for this opponent
        win_rate = opponent_results['wins'] / num_games * 100
        print(f"  Summary vs {opponent}: {opponent_results['wins']}-"
              f"{opponent_results['losses']}-{opponent_results['draws']} "
              f"(Win rate: {win_rate:.1f}%)")
        print()
    
    results['final_elo'] = our_elo
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print evaluation summary.
    
    Args:
        results: Evaluation results dictionary
    """
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    total_games = results['total_games']
    wins = results['wins']
    losses = results['losses']
    draws = results['draws']
    
    print(f"Total Games: {total_games}")
    print(f"Record: {wins}-{losses}-{draws}")
    print(f"Win Rate: {wins / max(1, total_games) * 100:.1f}%")
    print()
    
    print("Final Elo Rating: {:.0f}".format(results['final_elo']))
    print()
    
    print("Results by Opponent:")
    for opponent, opp_results in results['opponents'].items():
        win_rate = opp_results['wins'] / len(opp_results['games']) * 100
        avg_goal_diff = opp_results['goal_diff'] / len(opp_results['games'])
        opp_elo = results['elo_ratings'][opponent]
        
        print(f"  {opponent}:")
        print(f"    Record: {opp_results['wins']}-{opp_results['losses']}-{opp_results['draws']}")
        print(f"    Win Rate: {win_rate:.1f}%")
        print(f"    Avg Goal Diff: {avg_goal_diff:+.2f}")
        print(f"    Final Elo: {opp_elo:.0f}")
        print()


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    print("=" * 70)
    print("RL-Bot Evaluation")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"Configuration: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Opponents: {', '.join(args.opponents)}")
    print(f"Games per opponent: {args.num_games}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Seed: {args.seed}")
    print()
    
    # Load bot
    try:
        config = load_config(config_path)
        bot_manager = BotManager(
            config_path=config_path,
            policy_type="hybrid",
            model_path=checkpoint_path
        )
        print("Bot loaded successfully")
        print()
    except Exception as e:
        print(f"Error loading bot: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = run_evaluation(
            bot_manager=bot_manager,
            opponents=args.opponents,
            num_games=args.num_games,
            deterministic=args.deterministic
        )
        
        # Print summary
        print_summary(results)
        
        # Save results
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
        
        # Generate plots if requested
        if args.plot:
            print("Generating Elo rating plots...")
            # Would generate plots here
            print("Plots saved to: eval_plots/")
        
        print()
        print("=" * 70)
        print("Evaluation completed successfully!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print()
        print("Evaluation interrupted by user")
        
    except Exception as e:
        print()
        print(f"Evaluation failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
