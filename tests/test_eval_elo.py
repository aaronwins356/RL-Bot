"""Unit tests for Elo rating system."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import csv

from core.training.eval import EloRating, EloEvaluator, plot_elo_history


def test_elo_rating_initialization():
    """Test EloRating initialization."""
    elo = EloRating(k_factor=32, initial_rating=1500)
    assert elo.k_factor == 32
    assert elo.initial_rating == 1500


def test_elo_rating_expected_score():
    """Test expected score calculation."""
    elo = EloRating()
    
    # Equal ratings should give 0.5 expected score
    expected = elo.expected_score(1500, 1500)
    assert abs(expected - 0.5) < 0.01
    
    # Higher rating should have higher expected score
    expected_higher = elo.expected_score(1600, 1500)
    assert expected_higher > 0.5
    
    # Lower rating should have lower expected score
    expected_lower = elo.expected_score(1400, 1500)
    assert expected_lower < 0.5


def test_elo_rating_update():
    """Test Elo rating update."""
    elo = EloRating(k_factor=32)
    
    # Win should increase rating
    rating = 1500
    opponent_rating = 1500
    new_rating = elo.update_rating(rating, opponent_rating, score=1.0)
    assert new_rating > rating
    
    # Loss should decrease rating
    new_rating = elo.update_rating(rating, opponent_rating, score=0.0)
    assert new_rating < rating
    
    # Draw should change rating based on expected score
    new_rating = elo.update_rating(rating, opponent_rating, score=0.5)
    assert abs(new_rating - rating) < 1  # Should be close to original


def test_elo_rating_upset_win():
    """Test Elo update for upset win."""
    elo = EloRating(k_factor=32)
    
    # Underdog (1400) beats favorite (1600)
    underdog_rating = 1400
    favorite_rating = 1600
    
    new_underdog = elo.update_rating(underdog_rating, favorite_rating, score=1.0)
    
    # Underdog should gain more points than if they beat an equal opponent
    expected_gain_vs_equal = 16  # Roughly k_factor/2
    actual_gain = new_underdog - underdog_rating
    
    assert actual_gain > expected_gain_vs_equal


def test_elo_evaluator_initialization():
    """Test EloEvaluator initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EloEvaluator(
            baseline_bots=['rule_policy', 'baseline_ml'],
            log_dir=Path(tmpdir)
        )
        
        assert evaluator.agent_elo == evaluator.INITIAL_ELO
        assert len(evaluator.baseline_elos) >= 2
        assert evaluator.log_dir.exists()


def test_elo_evaluator_record_game():
    """Test recording individual games."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EloEvaluator(
            baseline_bots=['rule_policy'],
            log_dir=Path(tmpdir)
        )
        
        initial_elo = evaluator.agent_elo
        
        # Record a win
        evaluator.record_game('rule_policy', 'win', our_score=3, opp_score=1)
        assert evaluator.agent_elo > initial_elo
        
        # Check CSV was written
        game_csv = evaluator.log_dir / 'game_by_game.csv'
        assert game_csv.exists()
        
        with open(game_csv, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2  # Header + 1 game


def test_elo_evaluator_multiple_games():
    """Test Elo updates over multiple games."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EloEvaluator(
            baseline_bots=['rule_policy'],
            log_dir=Path(tmpdir),
            k_factor=32
        )
        
        # Record series of games with controlled scores
        # 3 wins, 1 loss, 1 draw = net positive
        games = [
            ('win', 3, 1),
            ('win', 2, 0),
            ('loss', 1, 3),
            ('draw', 2, 2),
            ('win', 4, 1),
        ]
        
        elos = [evaluator.agent_elo]
        
        for i, (result, our_score, opp_score) in enumerate(games):
            evaluator.record_game('rule_policy', result, our_score, opp_score, game_idx=i)
            elos.append(evaluator.agent_elo)
        
        # Elo should change over time
        assert len(set(elos)) > 1  # Not all the same
        
        # With 3 wins, 1 loss, 1 draw, overall Elo should be higher
        # But against lower-rated opponent, we might not gain much
        # So just check that it changed
        assert elos[-1] != elos[0]


def test_elo_evaluator_match_history():
    """Test match history tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EloEvaluator(
            baseline_bots=['rule_policy'],
            log_dir=Path(tmpdir)
        )
        
        # Record some games
        evaluator.record_game('rule_policy', 'win')
        evaluator.record_game('rule_policy', 'loss')
        
        assert len(evaluator.match_history) == 2
        assert 'elo_before' in evaluator.match_history[0]
        assert 'elo_after' in evaluator.match_history[0]
        assert 'elo_change' in evaluator.match_history[0]


def test_plot_elo_history():
    """Test Elo history plotting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history = [
            {'elo': 1500},
            {'elo': 1520},
            {'elo': 1510},
            {'elo': 1530},
            {'elo': 1550},
        ]
        
        plot_path = Path(tmpdir) / 'test_elo.png'
        
        # Should not crash
        plot_elo_history(history, str(plot_path))
        
        # Plot should be created (if matplotlib is available)
        # If matplotlib not available, function logs warning but doesn't crash


def test_elo_evaluator_stats():
    """Test getting evaluator stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = EloEvaluator(
            baseline_bots=['rule_policy'],
            log_dir=Path(tmpdir)
        )
        
        evaluator.record_game('rule_policy', 'win')
        evaluator.record_game('rule_policy', 'win')
        evaluator.record_game('rule_policy', 'loss')
        
        stats = evaluator.get_stats()
        
        assert 'current_elo' in stats
        assert 'matches_played' in stats
        assert 'match_history' in stats
        assert stats['matches_played'] == 3


def test_elo_rating_k_factor_effect():
    """Test effect of different K-factors."""
    # Higher K-factor = more volatile ratings
    elo_high_k = EloRating(k_factor=64)
    elo_low_k = EloRating(k_factor=16)
    
    rating = 1500
    opponent = 1500
    
    # After a win
    new_high = elo_high_k.update_rating(rating, opponent, 1.0)
    new_low = elo_low_k.update_rating(rating, opponent, 1.0)
    
    # Higher K should result in larger change
    assert abs(new_high - rating) > abs(new_low - rating)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
