
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

morph_kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], np.uint8)


def evolve(board, buff):
    cv2.copyMakeBorder(
        board, 1, 1, 1, 1, dst=buff, borderType=cv2.BORDER_WRAP
    )
    cv2.filter2D(buff, -1, morph_kernel, dst=buff)
    cv2.threshold(buff, 3, 0, cv2.THRESH_TOZERO_INV, dst=buff)
    cv2.scaleAdd(buff[1:-1, 1:-1], 1, board, dst=board)
    cv2.threshold(board, 2, 1, cv2.THRESH_BINARY, dst=board)
    return board


class GameBoard:
    def __init__(self, initial_board, noise_interval):
        self.board = initial_board.astype(np.uint8)
        self.board_vector = self.board.ravel()
        self.buff_bordered = cv2.copyMakeBorder(
            self.board, 1, 1, 1, 1, borderType=cv2.BORDER_DEFAULT
        )
        self.i = 0
        self.n_pixels = int(np.multiply(*self.board.shape))
        noise_proportion = 0.001
        self.noise_indices = np.zeros(
            int(self.n_pixels*noise_proportion), dtype=np.int32
        )
        self.noise_interval = noise_interval

    def tick(self):
        evolve(self.board, self.buff_bordered)
        if self.i >= self.noise_interval > 0:
            self.i = 0
            cv2.randu(self.noise_indices, 0, self.n_pixels)
            self.board_vector[self.noise_indices] = 1
        else:
            self.i += 1
        return self.board


def test_lifegame():
    glider_board = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    # return to the original state after 24 ticks (periodic boundary)
    gb = GameBoard(glider_board.copy(), 0)
    for _ in range(24):
        gb.tick()
    assert np.all(gb.board == glider_board)

def gui_runner(initial_board, delay=30, noise_interval=0):
    win_name = 'board'
    img_buffer = np.zeros_like(initial_board)
    cv2.convertScaleAbs(initial_board, dst=img_buffer, alpha=255)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img_buffer)
    cv2.waitKey()
    gb = GameBoard(initial_board, noise_interval)

    while cv2.waitKey(delay) < 0:
        cv2.convertScaleAbs(gb.tick(), dst=img_buffer, alpha=255)
        cv2.imshow(win_name, img_buffer)

def preprocess(image_path, height, width):
    if image_path:
        input_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if input_image is not None:
            cv2.threshold(
                input_image, 255, 1, cv2.THRESH_OTSU, dst=input_image
            )
            return input_image
    return np.random.randint(0, 2, (height, width), dtype=np.uint8)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--image', '-i', type=Path,
        help='input image'
    )
    parser.add_argument(
        '--height', '-H', type=int,
        default=780,
        help='board height (ignored when the input image is specified)'
    )
    parser.add_argument(
        '--width', '-W', type=int,
        default=1024,
        help='board width (ignored when the input image is specified)'
    )
    parser.add_argument(
        '--delay', '-d', type=int,
        default=30,
        help='delay between frames (ms)'
    )
    parser.add_argument(
        '--noise-interval', type=int,
        default=0,
        help='interval between noise additions (disabled when < 1)'
    )

    args = parser.parse_args()

    gui_runner(
        preprocess(args.image, args.height, args.width),
        delay=args.delay,
        noise_interval=args.noise_interval
    )

if __name__ == '__main__':
    main()
