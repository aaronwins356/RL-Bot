"""Unit tests for curriculum learning and self-play."""
import pytest
from pathlib import Path
import tempfile

from core.training.selfplay import SelfPlayManager, CurriculumStage
from core.training.curriculum import CurriculumManager


def test_curriculum_stage_dataclass():
    """Test CurriculumStage dataclass."""
    stage = CurriculumStage(
        name="Test Stage",
        stage_id=0,
        min_timestep=0,
        max_timestep=1000,
        game_mode="1v1",
        opponent_type="rule_policy"
    )
    
    assert stage.name == "Test Stage"
    assert stage.is_active(500)
    assert not stage.is_active(1500)


def test_selfplay_manager_initialization():
    """Test SelfPlayManager initialization."""
    manager = SelfPlayManager()
    
    assert manager is not None
    assert len(manager.stages) == 3  # Restricted to 3 stages: 1v1, 1v2, 2v2
    assert manager.current_stage_idx == 0


def test_selfplay_manager_stages():
    """Test curriculum stages are correctly defined."""
    manager = SelfPlayManager()
    
    # Check all 3 stages exist (restricted curriculum)
    assert len(manager.stages) == 3
    
    # Verify stage names
    stage_names = [stage.name for stage in manager.stages]
    assert stage_names == ["1v1", "1v2", "2v2"]
    
    # Check stage IDs are sequential
    for i, stage in enumerate(manager.stages):
        assert stage.stage_id == i


def test_selfplay_get_current_stage():
    """Test getting current stage based on timestep."""
    manager = SelfPlayManager()
    
    # Stage 0: 0 - 2M (1v1)
    stage = manager.get_current_stage(500_000)
    assert stage.stage_id == 0
    
    # Stage 1: 2M - 3.5M (1v2)
    stage = manager.get_current_stage(2_500_000)
    assert stage.stage_id == 1
    
    # Stage 2: 3.5M+ (2v2)
    stage = manager.get_current_stage(4_000_000)
    assert stage.stage_id == 2
    
    # Last stage should handle any timestep beyond max
    stage = manager.get_current_stage(100_000_000)
    assert stage.stage_id == 2


def test_selfplay_stage_transition():
    """Test stage transition detection."""
    manager = SelfPlayManager()
    
    # Should not transition within same stage
    should_transition, new_stage = manager.should_transition_stage(500_000)
    assert not should_transition
    
    # Simulate reaching stage boundary
    manager.current_stage_idx = 0
    should_transition, new_stage = manager.should_transition_stage(2_000_000)
    assert should_transition
    assert new_stage.stage_id == 1


def test_selfplay_opponent_pool():
    """Test opponent pool management."""
    manager = SelfPlayManager()
    
    # Start with empty pool
    assert len(manager.opponent_pool) == 0
    
    # Add opponents
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint1 = Path(tmpdir) / "checkpoint_1.pt"
        checkpoint1.touch()
        
        manager.add_opponent(checkpoint1, elo=1500, timestep=100_000)
        assert len(manager.opponent_pool) == 1
        
        checkpoint2 = Path(tmpdir) / "checkpoint_2.pt"
        checkpoint2.touch()
        
        manager.add_opponent(checkpoint2, elo=1600, timestep=200_000)
        assert len(manager.opponent_pool) == 2


def test_selfplay_select_opponent():
    """Test opponent selection logic."""
    manager = SelfPlayManager()
    
    # Get stage that requires opponent from pool
    stage = manager.get_current_stage(4_000_000)  # Stage 2 (2v2 selfplay)
    
    # With empty pool, should return fallback
    opponent = manager.select_opponent(stage)
    assert opponent is not None
    
    # Add opponent to pool
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "checkpoint.pt"
        checkpoint.touch()
        manager.add_opponent(checkpoint, elo=1500)
        
        # Should select from pool for selfplay stage
        opponent = manager.select_opponent(stage)
        assert opponent is not None


def test_selfplay_record_match():
    """Test match result recording."""
    manager = SelfPlayManager()
    
    # Record wins and losses
    manager.record_match_result(won=True, stage_id=0)
    manager.record_match_result(won=False, stage_id=0)
    manager.record_match_result(won=True, stage_id=0)
    
    assert manager.wins == 2
    assert manager.losses == 1
    assert manager.matches_played == 3
    
    # Check stage-specific stats
    stats = manager.get_stats()
    assert 'stage_stats' in stats
    assert 'stage_0_winrate' in stats['stage_stats']


def test_selfplay_stage_config():
    """Test getting stage configuration."""
    manager = SelfPlayManager()
    
    # Get config for stage 2 (2v2 selfplay)
    config = manager.get_stage_config(4_000_000)
    
    assert config['stage_id'] == 2
    assert config['game_mode'] == '2v2'
    assert config['opponent_type'] == 'selfplay'
    assert 'rotation_penalty_weight' in config
    assert config['rotation_penalty_weight'] > 0  # Should have rotation penalties


def test_selfplay_custom_stages():
    """Test custom curriculum stages from config (restricted to max 3 stages)."""
    custom_config = {
        'custom_stages': [
            {
                'name': 'Custom Stage 1',
                'min_timestep': 0,
                'max_timestep': 500_000,
                'game_mode': '1v1',
                'opponent_type': 'basic_script',
                'difficulty': 0.3
            },
            {
                'name': 'Custom Stage 2',
                'min_timestep': 500_000,
                'max_timestep': float('inf'),
                'game_mode': '3v3',
                'opponent_type': 'selfplay',
                'difficulty': 0.9
            }
        ]
    }
    
    manager = SelfPlayManager(custom_config)
    
    # Should be limited to 2 stages (as provided)
    assert len(manager.stages) == 2
    assert manager.stages[0].name == 'Custom Stage 1'
    assert manager.stages[1].name == 'Custom Stage 2'


def test_curriculum_manager_initialization():
    """Test CurriculumManager initialization."""
    config = {'aerial_focus': True}
    manager = CurriculumManager(config)
    
    assert manager is not None
    assert manager.aerial_focus
    assert len(manager.stages) > 0


def test_curriculum_stage_progression():
    """Test curriculum stage progression."""
    config = {'aerial_focus': False}
    manager = CurriculumManager(config)
    
    # Get stages at different timesteps
    stage_0 = manager.get_current_stage(500_000)
    stage_1 = manager.get_current_stage(3_000_000)
    stage_2 = manager.get_current_stage(6_000_000)
    
    # Should progress through stages
    assert stage_0.difficulty < stage_1.difficulty
    assert stage_1.difficulty < stage_2.difficulty


def test_curriculum_training_config():
    """Test curriculum training config generation."""
    config = {'aerial_focus': True}
    manager = CurriculumManager(config)
    
    # Get config for aerial-focused stage
    stage_config = manager.get_training_config(2_000_000)
    
    assert 'stage_name' in stage_config
    assert 'difficulty' in stage_config
    
    # Aerial stage should have aerial-specific settings
    if stage_config['aerial_focus']:
        assert 'spawn_height_range' in stage_config
        assert 'aerial_reward_weight' in stage_config


def test_opponent_update_frequency():
    """Test opponent update timing."""
    config = {'opponent_update_freq': 5}  # Use smaller number for testing
    manager = SelfPlayManager(config)
    
    # Should not update for first few calls
    assert not manager.should_update_opponent(0)
    assert not manager.should_update_opponent(1)
    assert not manager.should_update_opponent(2)
    assert not manager.should_update_opponent(3)
    
    # Should update on 5th call (0-indexed, so after 5 increments)
    result = manager.should_update_opponent(4)
    assert result
    
    # Counter should reset, so next call should not update
    assert not manager.should_update_opponent(5)


def test_selfplay_stats():
    """Test self-play statistics."""
    manager = SelfPlayManager()
    
    # Record some matches
    for i in range(10):
        won = i % 2 == 0  # Alternate wins/losses
        manager.record_match_result(won, stage_id=0)
    
    stats = manager.get_stats()
    
    assert stats['matches_played'] == 10
    assert stats['wins'] == 5
    assert stats['losses'] == 5
    assert abs(stats['win_rate'] - 0.5) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
