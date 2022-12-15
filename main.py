import torch
import modules
import trainer
import memory
import utils
from threading import Thread
import time


def main():
    device = torch.device('cpu')
    dtype = torch.float32
    if torch.cuda.is_available():
        print('CUDA available')
        device = torch.device('cuda')
    else:
        print('CUDA unavailable')

    size = (800, 450)
    nstack = 4

    # TODO: Load or create new model
    model = modules.A2C(size, nstack, (15, 2), actor_hidden_size=256, device=device, dtype=dtype)

    learner = trainer.Trainer(model, device=device, dtype=dtype)
    mem = memory.Memory()

    timer = utils.SimpleTimer(0.1)
    screen = utils.FrameGrabber(size=size)

    while True:
        # Training loop
        past_frames = torch.zeros((nstack, *size))

        start_game()  # TODO

        total_entropy = 0
        timer.start()
        while game_running:  # TODO: Get this from final reward?
            # Game loop
            past_frames[0] = screen.frame()
            action, log_prob, entropy, value = model.get_action(past_frames)
            send_action(action)  # TODO

            past_frames = torch.roll(past_frames, 1, dims=0)
            total_entropy += entropy

            timer.wait_and_continue()
            reward = get_reward()  # TODO
            mem.add(action, log_prob, value, reward)

        mem.entropy = total_entropy
        learner.train(mem)

def mainv2():
    device = torch.device('cpu')
    dtype = torch.float32
    if torch.cuda.is_available():
        print('CUDA available')
        device = torch.device('cuda')
    else:
        print('CUDA unavailable')

    size = (800, 450)
    nstack = 4

    # TODO: Load or create new model
    model = modules.A2C(size, nstack, (15, 2), actor_hidden_size=256, device=device, dtype=dtype)
    model.to(device=device, dtype=dtype)

    learner = trainer.Trainer(model, device=device, dtype=dtype)
    mem = memory.Memory()

    timer = utils.SimpleTimer(0.1)
    screen = utils.FrameGrabber(size=size)
    game = utils.GameManager()

    while True:
        # Training loop
        past_frames = torch.zeros((nstack, *size))

        print("Starting game...")

        # TODO: start_game()
        game.initloop(18, 'cpu', 18)  # No idea which character. Also, does this start the game?

        # Wait for game to start
        started = False
        while not started:
            img = screen.color_img()
            started = game.MatchStart(img)

        total_entropy = 0
        prev_actions = (14, 14)
        prev_stats = game.getcurrentstats()

        print('Game started, playing game')
        timer.start()
        while not game.MatchEnd():
            # Game loop
            past_frames[0] = screen.frame()
            actions, log_prob, entropy, value = model.get_action(past_frames)

            t = Thread(target=game.action, args=[*prev_actions, *actions])  # Might need to pass self
            t.start()

            past_frames = torch.roll(past_frames, 1, dims=0)
            total_entropy += entropy

            reward, prev_stats = game.get_reward(prev_stats, "Player 1")
            mem.add(actions, log_prob, value, reward)
            timer.wait_and_continue()

        print('Game finished, beginning training')
        start_time = time.time()

        mem.shift_rewards_left()
        mem.entropy = total_entropy
        learner.train(mem)

        end_time = time.time()
        print(f'Training iteration finished. Elapsed time: {end_time-start_time}')



if __name__ == '__main__':
    mainv2()
