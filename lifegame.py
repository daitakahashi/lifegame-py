
import itertools
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

morph_kernel = cv2.UMat(np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], np.uint8))


class GameBoard:
    def __init__(self, initial_board, noise_interval):
        self.board = cv2.UMat(initial_board.astype(np.uint8))
        self.board_size = initial_board.shape[:2]
        self.buff_bordered = cv2.copyMakeBorder(
            self.board, 1, 1, 1, 1, borderType=cv2.BORDER_DEFAULT
        )
        self.buff_center = cv2.UMat(
            self.buff_bordered, (1, self.board_size[0] + 1), (1, self.board_size[1] + 1)
        )
        self.i = 0
        self.n_pixels = int(np.multiply(*self.board_size))
        noise_proportion = 0.001
        self.noise_indices = np.zeros(
            int(np.ceil(self.n_pixels*noise_proportion)), dtype=np.int32
        )
        self.noise = np.zeros(self.board_size, dtype=np.uint8)
        self.noise_interval = noise_interval

    def tick(self):
        self.i += 1
        board = self.board
        buff = self.buff_bordered
        buff_center = self.buff_center
        cv2.copyMakeBorder(
            board, 1, 1, 1, 1, dst=buff, borderType=cv2.BORDER_WRAP
        )
        cv2.filter2D(buff, -1, morph_kernel, dst=buff)
        cv2.threshold(buff, 3, 0, cv2.THRESH_TOZERO_INV, dst=buff)
        cv2.scaleAdd(buff_center, 1, board, dst=board)
        cv2.threshold(board, 2, 1, cv2.THRESH_BINARY, dst=board)
        if self.i >= self.noise_interval > 0:
            self.i = 0
            cv2.randu(self.noise_indices, 0, self.n_pixels)
            self.noise[:] = 0
            self.noise.ravel()[self.noise_indices] = 1
            cv2.bitwise_or(
                cv2.UMat(self.noise), board, dst=board
            )
        return board

    def tick_np(self):
        self.tick()
        return self.board_np

    @property
    def board_np(self):
        return cv2.UMat.get(self.board)


def test_rules():
    def generate_problems(center='live', n=0):
        ixs = [0, 1, 2, 3, 5, 6, 7, 8]
        center_status = {'live': 1, 'dead': 0}
        problems = []
        for combn in itertools.combinations(ixs, n):
            mat = np.zeros((3, 3), dtype=np.uint8)
            mat.ravel()[np.array(combn, dtype=int)] = 1
            mat[1, 1] = center_status[center]
            problems.append(GameBoard(mat, 0))
        return problems
    def is_live(mat):
        return mat[1, 1] == 1
    def is_dead(mat):
        return mat[1, 1] == 0
    def check(center, surrounding_living_cells, test_next):
        return all([
            test_next(prob.tick_np())
            for prob in generate_problems(center, surrounding_living_cells)
        ])

    # surrounded by 0 living cells -> die
    assert check('dead', surrounding_living_cells=0, test_next=is_dead)
    assert check('live', surrounding_living_cells=0, test_next=is_dead)

    # surrounded by a living cell -> die
    assert check('dead', surrounding_living_cells=1, test_next=is_dead)
    assert check('live', surrounding_living_cells=1, test_next=is_dead)

    # surrounded by 2 living cells -> survive but no birth
    assert check('dead', surrounding_living_cells=2, test_next=is_dead)
    assert check('live', surrounding_living_cells=2, test_next=is_live)

    # surrounded by 3 living cells -> survive and birth
    assert check('dead', surrounding_living_cells=3, test_next=is_live)
    assert check('live', surrounding_living_cells=3, test_next=is_live)

    # surrounded by > 4 living cells -> die
    for n in range(4, 9):
        assert check('dead', surrounding_living_cells=n, test_next=is_dead)
        assert check('live', surrounding_living_cells=n, test_next=is_dead)


def test_glider():
    glider_board = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)
    # it returns to the original state after 24 ticks (periodic boundary)
    gb = GameBoard(glider_board.copy(), 0)
    for _ in range(24):
        gb.tick()
    assert np.all(gb.board_np == glider_board)


def test_noise():
    problem = np.zeros((5, 5), dtype=np.uint8)
    intervals = [1, 10, 100]
    for interval in intervals:
        gb = GameBoard(problem.copy(), interval)
        for _ in range(interval - 1):
            assert np.all(gb.tick_np() == problem)
        assert np.any(gb.tick_np() != problem)
    gb = GameBoard(problem.copy(), 0)
    for _ in range(1000):
        assert np.all(gb.tick_np() == problem)


def gui_runner(initial_board, delay=30, noise_interval=0):
    win_name = 'board'
    img_buffer = cv2.UMat(np.zeros_like(initial_board))
    cv2.convertScaleAbs(cv2.UMat(initial_board), dst=img_buffer, alpha=255)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img_buffer)
    cv2.waitKey()
    gb = GameBoard(initial_board, noise_interval)

    while cv2.waitKey(delay) < 0:
        cv2.convertScaleAbs(gb.tick(), dst=img_buffer, alpha=255)
        cv2.imshow(win_name, img_buffer)


def benchmark_runner(n_ticks, initial_board, delay=30, noise_interval=0):
    t0 = time.time()
    gb = GameBoard(initial_board, noise_interval)
    t1 = time.time()
    for _ in range(n_ticks):
        gb.tick()
    t2 = time.time()
    print('board initialization: {}s, time evolution ({} ticks): {}s'.format(
        t1 - t0, n_ticks, t2 - t1
    ))
    return
    


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
    parser.add_argument(
        '--benchmark', type=int,
        help='run non-gui benchmark with specified ticks'
    )

    args = parser.parse_args()

    if args.benchmark is None:
        gui_runner(
            preprocess(args.image, args.height, args.width),
            delay=args.delay,
            noise_interval=args.noise_interval
        )
    else:
        benchmark_runner(
            args.benchmark,
            preprocess(args.image, args.height, args.width),
            delay=args.delay,
            noise_interval=args.noise_interval
        )


if __name__ == '__main__':
    main()
