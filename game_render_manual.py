from agents import KeyboardAgent, PlayerV2
from pvz import config

agent_type = "Keyboard"

if __name__ == "__main__":
    print("🎮 Manual Plants vs Zombies Controls:")
    print("When prompted:")
    print("  'Do something (y/n)': Type 'y' to place a plant, 'n' to skip")
    print("  'Plant which': Choose plant type:")
    print("    0 = Sunflower (generates sun)")
    print("    1 = Peashooter (shoots zombies)")
    print("    2 = Wall-nut (blocks zombies)")
    print("    3 = Potato Mine (explosive defense)")
    print("  'Lane': Choose row (0-4, top to bottom)")
    print("  'Pos': Choose column (0-8, left to right)")
    print("=" * 50)

    if agent_type == "Keyboard":
        env = PlayerV2(render=True, max_frames = 500*config.FPS)
        agent = KeyboardAgent()

    env.play(agent)
    render_info = env.get_render_info()

    # Import and use the render function from game_render.py
    from game_render import render
    render(render_info)