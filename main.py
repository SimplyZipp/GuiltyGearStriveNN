import torch
import modules
import trainer
import memory
import utils
from threading import Thread
import time
from torch.cuda.amp import autocast


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
        while not game.MatchEnd():  # TODO: Get this from final reward?
            # Game loop
            past_frames[0] = screen.frame()
            action, log_prob, entropy, value = model.get_action(past_frames)
            #send_action(action)  # TODO

            past_frames = torch.roll(past_frames, 1, dims=0)
            total_entropy += entropy

            timer.wait_and_continue()
            #reward = get_reward()  # TODO
            reward=0
            mem.add(action, log_prob, value, reward)

        mem.entropy = total_entropy
        learner.train(mem)

def mainv2(load_path=None):
    device = torch.device('cpu')
    dtype = torch.float32
    if torch.cuda.is_available():
        print('CUDA available')
        device = torch.device('cuda')
    else:
        print('CUDA unavailable')

    size = (128, 128)
    nstack = 4

    # Load or create new model
    model = None
    model = modules.A2C(size, nstack, (2, 15), actor_hidden_size=256, device=device, dtype=dtype)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model.to(device=device, dtype=dtype)

    learner = trainer.Trainer(model, device=device, dtype=dtype)
    mem = memory.Memory()

    timer = utils.SimpleTimer(0.2)
    screen = utils.FrameGrabber(size=size)
    game = utils.GameManager()

    delay = 2
    print(f'Starting in {delay} seconds')
    time.sleep(delay)

    game.initloop(0, 'cpu', 0)  # Choose Sol as the character for both sides
    while True:
        # Training loop
        past_frames = torch.zeros((nstack, *size), device=device, dtype=dtype).detach()


        print("Starting game...")

        # Wait for game to start
        started = False
        while not started:
            img = screen.color_img()
            started = game.MatchStart(img)
            game.try_rematch()

        total_entropy = 0
        prev_actions = (14, 14)
        prev_stats = game.getcurrentstats()
        iterations = 0

        print('Game started, playing game')
        match_end, winlose = 0, 0
        timer.start()
        timeout = False #for the reward function
        start_time = time.time() #for keeping track of the match length
        while not match_end:
            iterations += 1
            # Game loop
            past_frames[0] = screen.frame()

            # Convert some calculations to half-precision to save memory
            with autocast():
                actions, log_prob, entropy, value = model.get_action(past_frames)

            actions = (actions[0].item(), actions[1].item())

            t = Thread(target=game.action, args=[*prev_actions, *actions])  # Might need to pass self
            t.start()

            past_frames = torch.roll(past_frames, 1, dims=0).detach()
            total_entropy += entropy
            match_end, winlose = game.MatchEnd() #moved back here so it doesn't interfere with the case where too much time passes
            if time.time() - start_time > 99: #may replace with a memory address in match_end later
                timeout = True
                match_end = True
            reward, prev_stats = game.get_reward(prev_stats, "Player 1", timeout)
            mem.add(actions, log_prob.mean(), value, reward)
            prev_actions = actions

            if iterations % 200 == 0:
                # Do some training to alleviate memory usage
                iterations = 0
                time.sleep(0.1)
                final_reward, _ = game.get_reward(prev_stats, "Player 1", timeout)
                quick_train(learner, mem, total_entropy, final_reward)
                total_entropy = 0

            timer.wait_and_continue()

        if winlose == 0:
            winlose = -1
        quick_train(learner, mem, total_entropy, winlose * 20)
        game.reset()


def quick_train(learner, mem, total_entropy, final_reward=0):
    print('Beginning training')
    start_time = time.time()

    mem.shift_rewards_left()
    mem.entropy = total_entropy
    learner.train(mem)

    end_time = time.time()
    print(f'Training iteration finished. Elapsed time: {end_time - start_time}')



if __name__ == '__main__':
    mainv2()
