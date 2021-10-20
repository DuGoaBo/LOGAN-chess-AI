import chess
import torch
import random
import copy
from LOGAN_module import *
from tensorize_board import *

gamma = 0.95
epsilon = 0.1
learning_rate = 0.01
epochs = 2000
episode_length = 300
L_train = LOGAN_module()
L_target = LOGAN_module()

def generate_training_episode(gamma, epsilon):
    ret_x = torch.Tensor(episode_length, 1, 8, 8)
    ret_t = torch.Tensor(episode_length, 1)
    b = chess.Board()
    for i in range(episode_length):
        moves = list(b.legal_moves)
        best_move = None
        best_move_evaluation = float("inf")
        for move in moves:
            b.push(move)
            move_evaluation = L_target(tensorize_board(b)).item()
            if move_evaluation < best_move_evaluation:
                best_move = move
                best_move_evaluation = move_evaluation
            b.pop()
        ret_x[i] = tensorize_board(b)
        ret_t[i] = torch.tensor(calculate_material(b, ret_x[i]) +  gamma * -1 * best_move_evaluation)
        r = random.random()
        next_move = moves[0]
        if r < epsilon or not best_move:
            next_move = random.choice(moves)
        else:
            next_move = best_move
        b.push(next_move)
        if b.is_game_over():
            if b.outcome().winner == True:
                ret_t[i] += 100
            elif b.outcome().winner == False:
                ret_t[i] -= 100
            print(b.outcome())
            return ret_x[:i], ret_t[:i]
    print("game ended after {} moves".format(episode_length))
    return ret_x, ret_t

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(L_train.parameters(), lr = learning_rate)
for epoch in range(epochs):
    print("epoch: " + str(epoch))
    print("epsilon: " + str(epsilon))
    train_x, train_t = generate_training_episode(gamma, epsilon)
    optimizer.zero_grad()
    outputs = L_train(train_x)
    loss = criterion(outputs, train_t)
    loss.backward()
    print(loss)
    if loss > 200:
        L_train = torch.load("L")
        continue
    optimizer.step()
    if epoch % 100 == 0:
        L_target = copy.deepcopy(L_train)
        torch.save(L_train, "L")
    epsilon = epsilon * 0.999
    print("\n")
 
