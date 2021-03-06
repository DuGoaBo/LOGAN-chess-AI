import torch
import chess
from tensorize_board import *
from LOGAN_module import *

L = torch.load("L")
print(list(L.named_parameters()))

b = chess.Board()
while True:
    print(str(b) + "\n")
    print("enter your move as a san")
    move = input("")
    try:
        b.push_san(move)
    except ValueError:
        print("that move is not valid")
        continue
    print(L(tensorize_board(b)))
    computer_moves = list(b.legal_moves)
    best_computer_move = computer_moves[0]
    best_computer_move_eval = float("inf")
    for move in computer_moves:
        b.push(move)
        move_eval = L(tensorize_board(b)).item()
        if move_eval < best_computer_move_eval:
            best_computer_move = move
            best_computer_move_eval = move_eval
        b.pop()
    print(best_computer_move_eval)
    b.push(best_computer_move)
