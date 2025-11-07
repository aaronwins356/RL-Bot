"""Unit tests for hierarchical RL/IL pipeline components."""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFastAerialTiming(unittest.TestCase):
    """Test fast aerial timing constraints from §3.1."""
    
    def test_inter_jump_frames_range(self):
        """Test that inter-jump frames stay within [8, 14] frames."""
        from core.llc import FastAerialHelper
        
        config = {
            'inter_jump_frames': [10, 12],
        }
        helper = FastAerialHelper(config)
        
        # Check bounds
        self.assertGreaterEqual(helper.inter_jump_frames_min, 8)
        self.assertLessEqual(helper.inter_jump_frames_max, 14)
    
    def test_pitch_up_range(self):
        """Test that pitch up stays within [0.6, 0.9]."""
        from core.llc import FastAerialHelper
        
        config = {
            'pitch_up': [0.6, 0.9],
        }
        helper = FastAerialHelper(config)
        
        self.assertGreaterEqual(helper.pitch_up_min, 0.6)
        self.assertLessEqual(helper.pitch_up_max, 0.9)
    
    def test_should_second_jump(self):
        """Test second jump timing logic."""
        from core.llc import FastAerialHelper
        
        config = {
            'inter_jump_frames': [10, 12],
        }
        helper = FastAerialHelper(config)
        
        # Set first jump
        helper.first_jump_frame = 100
        
        # Too early
        self.assertFalse(helper.should_second_jump(105))
        
        # In window
        self.assertTrue(helper.should_second_jump(110))
        self.assertTrue(helper.should_second_jump(112))
        
        # Too late
        self.assertFalse(helper.should_second_jump(115))


class TestFlipResetDetector(unittest.TestCase):
    """Test flip reset detector from §3.5."""
    
    def test_four_wheel_contact_window(self):
        """Test that 4-wheel contact window is ≤ 20ms."""
        from core.llc import FlipResetDetector
        
        config = {
            'contact_window_ms': 20,
        }
        detector = FlipResetDetector(config)
        
        self.assertLessEqual(detector.contact_window_ms, 20)
    
    def test_max_relative_velocity(self):
        """Test max relative velocity constraint."""
        from core.llc import FlipResetDetector
        
        config = {
            'max_relative_velocity': 150.0,
        }
        detector = FlipResetDetector(config)
        
        self.assertEqual(detector.max_relative_velocity, 150.0)


class TestMustyConditions(unittest.TestCase):
    """Test musty trigger conditions from §3.6."""
    
    def test_nose_angle_range(self):
        """Test that musty requires nose angle ≥ 60° and ball above roof ≥ 30uu."""
        from core.skill_programs.musty import SP_Musty
        
        config = {
            'nose_angle': [60, 110],
            'min_boost': 30,
        }
        sp = SP_Musty(config)
        
        # Check angle range
        self.assertEqual(sp.nose_up_angle_range[0], 60)
        self.assertEqual(sp.nose_up_angle_range[1], 110)
    
    def test_min_boost_requirement(self):
        """Test that musty requires boost ≥ 30."""
        from core.skill_programs.musty import SP_Musty
        
        config = {
            'min_boost': 30,
        }
        sp = SP_Musty(config)
        
        self.assertGreaterEqual(sp.min_boost, 30)


class TestBreeziOscillation(unittest.TestCase):
    """Test breezi oscillation amplitude from §3.6."""
    
    def test_roll_amplitude_range(self):
        """Test that breezi maintains lateral oscillation amplitude within [0.12, 0.25]."""
        from core.skill_programs.breezi import SP_Breezi
        
        config = {
            'roll_amp': [0.12, 0.25],
            'roll_freq_hz': [5, 9],
        }
        sp = SP_Breezi(config)
        
        self.assertGreaterEqual(sp.roll_amp[0], 0.12)
        self.assertLessEqual(sp.roll_amp[1], 0.25)
    
    def test_roll_frequency_range(self):
        """Test that breezi maintains frequency at 5-9 Hz."""
        from core.skill_programs.breezi import SP_Breezi
        
        config = {
            'roll_amp': [0.12, 0.25],
            'roll_freq_hz': [5, 9],
        }
        sp = SP_Breezi(config)
        
        self.assertGreaterEqual(sp.roll_freq_hz[0], 5)
        self.assertLessEqual(sp.roll_freq_hz[1], 9)


class TestSkillProgramBase(unittest.TestCase):
    """Test skill program base class."""
    
    def test_timeout_checking(self):
        """Test timeout checking logic."""
        from core.skill_programs.base import SkillProgram
        
        class DummySP(SkillProgram):
            def reset(self, obs): pass
            def policy(self, obs): pass
            def should_terminate(self, obs): return False
            def get_fallback(self, obs): return None
        
        config = {'timeout': 2.0}
        sp = DummySP(config)
        
        # Not started
        self.assertFalse(sp.check_timeout(0.0))
        
        # Activate
        sp.activate({}, 0.0)
        
        # Within timeout
        self.assertFalse(sp.check_timeout(1.0))
        
        # Timeout reached
        self.assertTrue(sp.check_timeout(2.0))
        self.assertTrue(sp.check_timeout(2.5))


class TestRiskScorer(unittest.TestCase):
    """Test risk scorer from §8."""
    
    def test_risk_score_range(self):
        """Test that risk score is always in [0, 1]."""
        from core.opportunity_detector.risk_scorer import RiskScorer
        
        config = {
            'safe_time_threshold': 45.0,
        }
        scorer = RiskScorer(config)
        
        # Low risk scenario
        obs_low_risk = {
            'score_diff': 2,
            'time_remaining': 120.0,
            'is_last_defender': False,
            'ball_position': [0, 2000, 500],
            'our_goal_y': -5120,
        }
        
        risk = scorer.compute_risk_score(obs_low_risk)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        
        # High risk scenario
        obs_high_risk = {
            'score_diff': -1,
            'time_remaining': 20.0,
            'is_last_defender': True,
            'ball_position': [0, -4500, 200],
            'our_goal_y': -5120,
        }
        
        risk = scorer.compute_risk_score(obs_high_risk)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        self.assertGreater(risk, 0.5)  # Should be high risk
    
    def test_flashy_attempt_requirements(self):
        """Test that should_attempt_flashy checks all conditions."""
        from core.opportunity_detector.risk_scorer import RiskScorer
        
        config = {}
        scorer = RiskScorer(config)
        
        # Valid scenario
        obs_valid = {
            'possession_prob': 0.7,
            'own_eta_to_ball': 0.8,
            'opponent_closest_eta': 1.2,
            'boost': 50,
            'score_diff': 1,
            'time_remaining': 60.0,
            'is_last_defender': False,
            'ball_position': [0, 2000, 500],
            'our_goal_y': -5120,
        }
        
        self.assertTrue(scorer.should_attempt_flashy(obs_valid))
        
        # Invalid: low boost
        obs_low_boost = obs_valid.copy()
        obs_low_boost['boost'] = 20
        self.assertFalse(scorer.should_attempt_flashy(obs_low_boost))
        
        # Invalid: last defender in defensive third
        obs_defensive = obs_valid.copy()
        obs_defensive['is_last_defender'] = True
        obs_defensive['ball_position'] = [0, -4500, 200]
        self.assertFalse(scorer.should_attempt_flashy(obs_defensive))


class TestHierarchicalRewards(unittest.TestCase):
    """Test reward shaping from §5."""
    
    def test_contact_reward(self):
        """Test contact reward value."""
        from core.training.hierarchical_rewards import HierarchicalRewardShaper
        
        config = {
            'contact': 1.0,
        }
        shaper = HierarchicalRewardShaper(config)
        
        self.assertEqual(shaper.r_contact, 1.0)
    
    def test_style_bonus_values(self):
        """Test style bonus values match spec."""
        from core.training.hierarchical_rewards import HierarchicalRewardShaper
        
        config = {
            'ceiling_bonus': 0.6,
            'musty_bonus': 0.5,
            'breezi_bonus': 0.5,
            'doubletap_bonus': 0.7,
            'flipreset_goal_bonus': 1.0,
        }
        shaper = HierarchicalRewardShaper(config)
        
        self.assertEqual(shaper.r_ceiling_shot, 0.6)
        self.assertEqual(shaper.r_musty, 0.5)
        self.assertEqual(shaper.r_breezi, 0.5)
        self.assertEqual(shaper.r_double_tap, 0.7)
        self.assertEqual(shaper.r_flip_reset_goal, 1.0)
    
    def test_cost_values(self):
        """Test cost values are negative."""
        from core.training.hierarchical_rewards import HierarchicalRewardShaper
        
        config = {
            'boost_cost': -0.004,
            'whiff_cost': -0.2,
            'risky_flashy_cost': -0.15,
        }
        shaper = HierarchicalRewardShaper(config)
        
        self.assertLess(shaper.c_boost, 0)
        self.assertLess(shaper.c_whiff, 0)
        self.assertLess(shaper.c_risky_flashy, 0)


class TestDrillEvaluator(unittest.TestCase):
    """Test drill evaluator from §6."""
    
    def test_gate_thresholds(self):
        """Test gate thresholds match spec."""
        from core.training.drill_evaluator import DrillEvaluator
        
        config = {
            'fast_aerial': {'threshold': 0.88},
            'double_tap': {'threshold': 0.40},
            'ceiling': {'threshold': 0.60},
            'flip_reset': {'threshold': 0.35, 'convert': 0.20},
            'musty': {'threshold': 0.30},
            'breezi': {'threshold': 0.30},
        }
        
        evaluator = DrillEvaluator(config)
        
        self.assertEqual(evaluator.gates['fast_aerial']['threshold'], 0.88)
        self.assertEqual(evaluator.gates['double_tap']['threshold'], 0.40)
        self.assertEqual(evaluator.gates['ceiling']['threshold'], 0.60)
        self.assertEqual(evaluator.gates['flip_reset']['threshold'], 0.35)
        self.assertEqual(evaluator.gates['musty']['threshold'], 0.30)
        self.assertEqual(evaluator.gates['breezi']['threshold'], 0.30)


if __name__ == '__main__':
    unittest.main()
