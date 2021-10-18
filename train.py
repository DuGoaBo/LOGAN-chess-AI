import chess
import torch
import random
import copy
from LOGAN_module import *
from tensorize_board import *

gamma = 0.95
epsilon = 0.9
learning_rate = 0.01
epochs = 100
episode_length = 200
L_train = LOGAN_module()
L_target = LOGAN_module()

def generate_training_episode(gamma, epsilon):
    ret_x = torch.Tensor(episode_length, 67)
    ret_t = torch.Tensor(episode_length, 1)
    b = chess.Board()
    for i in range(episode_length):
        moves = list(b.legal_moves)
        best_move = moves[0]
        best_move_evaluation = float("inf")
        for move in moves:
            b.push(move)
            move_evaluation = L_target(tensorize_board(b)).item()
            if move_evaluation < best_move_evaluation:
                best_move = move
                best_move_evaluation = move_evaluation
            b.pop()
        ret_x[i] = tensorize_board(b)
        ret_t[i] = torch.tensor(gamma * -1 * best_move_evaluation)
        r = random.random()
        if r < epsilon:
            next_move = random.choice(moves)
        else:
            next_move = best_move
        b.push(next_move)
        if b.is_game_over():
            if b.outcome().winner == True:
                ret_t[i] += 1
            elif b.outcome().winner == False:
                ret_t[i] -= 1
            print(b.outcome())
            return ret_x[:i], ret_t[:i]
    print("game ended after 100 moves")
    return ret_x, ret_t


criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(L_train.parameters(), lr = learning_rate)
for epoch in range(epochs):
    train_x, train_t = generate_training_episode(gamma, epsilon)
    optimizer.zero_grad()
    outputs = L_train(train_x)
    loss = criterion(outputs, train_t)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        L_target = copy.deepcopy(L_train)
    epsilon = epsilon ** 1.01
 
torch.save(L_train, "L")
