import torch
import modules
import trainer
import memory
import utils


def main():
    device = 'cpu'
    dtype = torch.float32
    if torch.cuda.is_available():
        print('CUDA available')
        device = 'cuda'
    else:
        print('CUDA unavailable')

    size = (800, 450)
    nstack = 4

    # TODO: Load or create new model
    model = modules.A2C(size, nstack, (15, 2), actor_hidden_size=256, device=device, dtype=dtype)

    learner = trainer.Trainer(model, device=device, dtype=dtype)
    mem = memory.Memory()

    timer = utils.SimpleTimer(0.2)
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

            past_frames = torch.roll(past_frames, 1, dim=0)
            total_entropy += entropy

            timer.wait_and_continue()
            reward = get_reward()  # TODO
            mem.add(action, log_prob, value, reward)

        mem.entropy = total_entropy
        learner.train(mem)



if __name__ == '__main__':
    main()
