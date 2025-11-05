# Quick Start Guide - WinYour1s SSL Bot

Get up and running with the SSL-level bot in minutes!

## Prerequisites

- **Rocket League** (Steam version recommended)
- **Python 3.9+** installed
- **RLBot Framework** installed

## Installation

### 1. Install RLBot

```bash
pip install rlbot
```

Or download the RLBot GUI from: https://rlbot.org/

### 2. Clone This Repository

```bash
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Note: If you get an error about PyTorch version, install manually:
```bash
pip install torch numpy
```

## Running the Bot

### Method 1: RLBot GUI (Recommended)

1. Launch RLBot GUI
2. Click "Add" → "Browse for bot config"
3. Navigate to this folder and select `bot.cfg`
4. Click "Start Match"
5. Watch your bot play!

### Method 2: Command Line

```bash
# From the RL-Bot directory
rlbot gui
```

Then follow Method 1 steps above.

## Quick Test

Want to see the bot in action immediately?

1. Open RLBot GUI
2. Add "WinYour1s" bot (this bot)
3. Add "Psyonix Allstar" as opponent
4. Set match type to "1v1" 
5. Start match
6. Watch the SSL systems in action!

## What to Expect

### Current Capabilities (Diamond 2 - Champ 2 Level)

✅ **Fast Reactions**: 30 FPS decision-making
✅ **Ball Reading**: Predicts ball trajectory 4 seconds ahead
✅ **Boost Control**: Strategic boost collection and denial
✅ **Speedflip Kickoffs**: Fast, reliable kickoff technique
✅ **Basic Mechanics**: Wavedash, halfflip, fast aerials
✅ **Smart Positioning**: Strategic decision-making

⚠️ **Work in Progress**:
- Shadow defense (implemented but needs integration)
- Dribble control (coming soon)
- Fake challenges (implemented but needs integration)
- Wall play and ceiling shots (future)

### Performance Notes

- **Tick Skip = 4**: Bot makes decisions every 4 game ticks (30 FPS)
- **Model Size**: ~1.2MB pretrained neural network
- **Systems**: Ball prediction, boost management, utility-based decisions

## Customization

### Adjusting Bot Behavior

Edit `bot.py` to tune behavior:

```python
# Make bot more aggressive
self.tick_skip = 3  # Faster reactions (40 FPS)

# Make bot more defensive  
# Adjust scores in decision/utility_system.py
```

### Changing Appearance

Edit `appearance.cfg` to customize car appearance:
- Car body
- Decal
- Colors
- Wheels
- Boost trail

## Troubleshooting

### Bot Not Appearing in RLBot GUI

**Solution**: 
1. Check that `bot.cfg` exists in this folder
2. Try "Scan for bots" in RLBot GUI
3. Restart RLBot GUI

### Import Errors

**Solution**:
```bash
pip install --upgrade rlbot numpy torch
```

### Bot Not Moving

**Solution**:
1. Check Rocket League is running
2. Check RLBot GUI shows "Connected"
3. Try starting a match from RLBot GUI

### Performance Issues (Low FPS)

**Solution**: Edit `bot.py`:
```python
self.tick_skip = 6  # Slower but more stable (20 FPS)
```

Or in `util/ball_prediction.py`:
```python
self.prediction_horizon = 2.0  # Reduce from 4.0 to 2.0 seconds
```

### Bot Playing Poorly

The bot is still in development! Current version focuses on:
- Core systems implementation
- Strategic decision-making
- Mechanical reliability

For better performance:
1. Read `IMPLEMENTATION_GUIDE.md` to integrate all systems
2. Tune parameters in `decision/utility_system.py`
3. Test against various opponents to identify weaknesses

## Understanding the Bot

### Decision-Making Flow

```
Game Tick
    ↓
Update Game State
    ↓
Ball Prediction (4s ahead)
    ↓
Boost Management (track pads)
    ↓
[If Kickoff] → Speedflip Sequence
    ↓
[If Mechanical Sequence Active] → Execute Sequence
    ↓
[Otherwise] → Neural Network Decision
    ↓
Update Controls → Send to Game
```

### Key Files

- `bot.py`: Main bot logic, game loop
- `agent.py`: Neural network inference
- `util/ball_prediction.py`: Ball trajectory prediction
- `util/boost_manager.py`: Boost pad tracking and strategy
- `decision/utility_system.py`: Strategic behavior selection
- `sequences/`: Mechanical sequences (speedflip, wavedash, etc.)

## Performance Metrics

Track bot performance:

### In-Game Observations
- Does bot win kickoffs? (Target: >60%)
- Does bot maintain boost? (Target: >70% time with >50 boost)
- Does bot make good challenges?
- Does bot recover quickly after aerials?

### After Match
Check match stats:
- Score differential
- Shot accuracy
- Ball touches
- Boost usage

## Next Steps

### For Testing
1. Run bot against progressively harder opponents
2. Note specific situations where bot struggles
3. Review `SSL_UPGRADE_ANALYSIS.md` for detailed explanations
4. Tune parameters based on observations

### For Development
1. Read `IMPLEMENTATION_GUIDE.md` for integration instructions
2. Implement full utility system integration
3. Add shadow defense behavior to control flow
4. Test and tune each system individually

### For Learning
1. Watch replays with bot
2. Compare to SSL player replays
3. Identify behavior gaps
4. Iterate on improvements

## Community & Support

### Resources
- **RLBot Discord**: https://discord.gg/rlbot
- **RLBot Reddit**: r/RocketLeagueBots
- **This Repository**: Open issues for bugs or questions

### Contributing
Want to help improve the bot?
1. Test and report issues
2. Suggest improvements
3. Contribute code (see IMPLEMENTATION_GUIDE.md)
4. Share performance data

## Advanced Usage

### Training New Model (Optional)

To train a new neural network:

1. Install RLGym: `pip install rlgym stable-baselines3`
2. Create training script (see `SSL_UPGRADE_ANALYSIS.md` section 6.4)
3. Train for 10M+ timesteps
4. Replace `model.p` with new model
5. Test and iterate

### Performance Profiling

To identify performance bottlenecks:

```python
import cProfile
import pstats

# Add to bot.py
profiler = cProfile.Profile()
profiler.enable()

# ... bot code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

### Debugging

Enable debug output in `bot.py`:

```python
# Add to get_output() method
print(f"Behavior: {behavior.value}")
print(f"Boost: {player.boost_amount:.2f}")
print(f"Ball distance: {np.linalg.norm(ball.position - car.position):.0f}")
```

## FAQ

**Q: Why is my bot not as good as SSL players?**
A: Bot is work-in-progress! Current version has core systems but needs full integration and tuning.

**Q: Can I use this bot in ranked matches?**
A: No! RLBot is for private matches and training only. Using bots in ranked violates Rocket League TOS.

**Q: How was the bot trained?**
A: Original model trained with RLGym using PPO algorithm. SSL systems added on top.

**Q: Can I modify the bot?**
A: Yes! It's open source. See IMPLEMENTATION_GUIDE.md for how to customize and extend.

**Q: Will this reach SSL level?**
A: With continued development, tuning, and possibly retraining the NN, yes! Current systems provide the foundation.

**Q: How do I report a bug?**
A: Open an issue on GitHub with description, steps to reproduce, and any error messages.

## Success Criteria

Your bot is working correctly if:

✅ Bot successfully joins matches via RLBot GUI
✅ Bot executes speedflip kickoffs
✅ Bot moves purposefully toward ball/boost
✅ Bot makes aerials when ball is high
✅ Bot doesn't get stuck or freeze
✅ Bot maintains reasonable boost levels
✅ Bot makes saves and scores goals

Don't worry if:
❓ Bot loses to higher-level bots (expected)
❓ Bot makes some mistakes (normal)
❓ Bot doesn't use all mechanics yet (work in progress)

## Getting Help

1. **Check the logs**: RLBot GUI shows error messages
2. **Read the docs**: `README.md`, `SSL_UPGRADE_ANALYSIS.md`, `IMPLEMENTATION_GUIDE.md`
3. **Test components**: Run individual systems to isolate issues
4. **Ask for help**: RLBot Discord community is helpful!

---

**Ready to see SSL-level systems in action?** Fire up RLBot GUI and start a match! 🚀

For detailed technical information, see:
- `README.md` - Project overview and architecture
- `SSL_UPGRADE_ANALYSIS.md` - Comprehensive technical analysis  
- `IMPLEMENTATION_GUIDE.md` - Integration and testing instructions
