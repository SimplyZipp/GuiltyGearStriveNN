import torch
# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    if torch.cuda.is_available():
        print('CUDA available')
    else:
        print('CUDA unavailable')
    print('Not implemented')


if __name__ == '__main__':
    main()
