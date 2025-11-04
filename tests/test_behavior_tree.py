import pytest
from typing import List, Tuple
import numpy as np
from unittest.mock import Mock, patch

from rlbot_pro.planning.selector.tree import (
    BehaviorTree, Node, NodeStatus, ShotOpportunity,
    Selector, Sequence
)

@pytest.fixture
def mock_nodes() -> List[Mock]:
    return [Mock(spec=Node) for _ in range(3)]

def test_selector_all_fail(mock_nodes):
    """Test Selector when all nodes fail"""
    for node in mock_nodes:
        node.tick.return_value = NodeStatus.FAILURE
    
    selector = Selector(mock_nodes)
    assert selector.tick(Mock()) == NodeStatus.FAILURE
    
    for node in mock_nodes:
        node.tick.assert_called_once()

def test_selector_first_succeeds(mock_nodes):
    """Test Selector stops at first success"""
    mock_nodes[0].tick.return_value = NodeStatus.SUCCESS
    
    selector = Selector(mock_nodes)
    assert selector.tick(Mock()) == NodeStatus.SUCCESS
    
    mock_nodes[0].tick.assert_called_once()
    mock_nodes[1].tick.assert_not_called()
    mock_nodes[2].tick.assert_not_called()

def test_selector_running_state(mock_nodes):
    """Test Selector with running node"""
    mock_nodes[1].tick.return_value = NodeStatus.RUNNING
    mock_nodes[0].tick.return_value = NodeStatus.FAILURE
    
    selector = Selector(mock_nodes)
    assert selector.tick(Mock()) == NodeStatus.RUNNING
    
    mock_nodes[0].tick.assert_called_once()
    mock_nodes[1].tick.assert_called_once()
    mock_nodes[2].tick.assert_not_called()

def test_sequence_success(mock_nodes):
    """Test Sequence completes successfully"""
    for node in mock_nodes:
        node.tick.return_value = NodeStatus.SUCCESS
        
    sequence = Sequence(mock_nodes)
    assert sequence.tick(Mock()) == NodeStatus.SUCCESS
    
    for node in mock_nodes:
        node.tick.assert_called_once()
        node.reset.assert_called_once()

def test_sequence_failure(mock_nodes):
    """Test Sequence fails and resets"""
    mock_nodes[0].tick.return_value = NodeStatus.SUCCESS
    mock_nodes[1].tick.return_value = NodeStatus.FAILURE
    
    sequence = Sequence(mock_nodes)
    assert sequence.tick(Mock()) == NodeStatus.FAILURE
    
    mock_nodes[0].tick.assert_called_once()
    mock_nodes[1].tick.assert_called_once()
    mock_nodes[2].tick.assert_not_called()
    
    for node in mock_nodes:
        node.reset.assert_called_once()

def test_sequence_running(mock_nodes):
    """Test Sequence with running node"""
    mock_nodes[0].tick.return_value = NodeStatus.SUCCESS
    mock_nodes[1].tick.return_value = NodeStatus.RUNNING
    
    sequence = Sequence(mock_nodes)
    assert sequence.tick(Mock()) == NodeStatus.RUNNING
    
    # Run again - should continue from previous state
    assert sequence.tick(Mock()) == NodeStatus.RUNNING
    
    assert mock_nodes[0].tick.call_count == 1
    assert mock_nodes[1].tick.call_count == 2
    mock_nodes[2].tick.assert_not_called()

def test_shot_opportunity_evaluation():
    """Test shot opportunity evaluation"""
    shot = ShotOpportunity(
        probability=0.8,
        angle_to_goal=45.0,
        shot_speed=100.0,
        boost_required=30.0,
        pressure_level=0.2
    )
    
    # Test shot is viable
    assert shot.probability > 0.5
    assert 0 <= shot.angle_to_goal <= 90
    assert shot.shot_speed > 0
    assert 0 <= shot.boost_required <= 100
    assert 0 <= shot.pressure_level <= 1

def test_tree_config_loading():
    """Test behavior tree configuration loading"""
    mock_config = {
        'planner': {
            'mechanics': {'aerial': True},
            'aggression': {'base_level': 0.7}
        }
    }
    
    with patch('yaml.safe_load') as mock_load:
        mock_load.return_value = mock_config
        tree = BehaviorTree('dummy_path')
        
        assert tree.config['mechanics']['aerial'] is True
        assert tree.config['aggression']['base_level'] == 0.7

def test_node_reset_cascading():
    """Test reset cascades through node hierarchy"""
    nodes = [Mock(spec=Node) for _ in range(3)]
    sequence = Sequence(nodes)
    selector = Selector([sequence, Mock(spec=Node)])
    
    selector.reset()
    
    for node in nodes:
        node.reset.assert_called_once()

def test_tree_integration():
    """Test complete behavior tree integration"""
    with patch('yaml.safe_load') as mock_load:
        mock_load.return_value = {'planner': {
            'mechanics': {'aerial': True},
            'aggression': {'base_level': 0.7}
        }}
        
        tree = BehaviorTree('dummy_path')
        game_state = Mock()
        
        # Test normal gameplay scenario
        game_state.boost = 100
        game_state.ball_position = np.array([100, 0, 0])
        assert tree.tick(game_state) in [NodeStatus.RUNNING, NodeStatus.SUCCESS]
        
        # Test defensive scenario
        game_state.boost = 10
        game_state.ball_position = np.array([-1000, 0, 0])
        assert tree.tick(game_state) == NodeStatus.RUNNING