import chess
import torch
import random
import copy
from LOGAN_module import *
from tensorize_board import *

gamma = 0.99
epsilon = 0.9
learning_rate = 0.01
epochs = 30000
episode_length = 300
L_train = LOGAN_module()
L_target = LOGAN_module()

def generate_training_episode(gamma, epsilon):
    ret_x = torch.Tensor(episode_length, 1, 8, 8)
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
        ret_t[i] = torch.tensor(gamma**2 * calculate_material(b, ret_x[i]) +  gamma * -1 * best_move_evaluation)
        r = random.random()
        next_move = moves[0]
        if r < epsilon:
            r2 = random.random()
            if r2 < epsilon:
                random.shuffle(moves)
                for move in moves:
                    if b.is_capture(move):
                        next_move = move
                        break
            else:
                next_move = random.choice(moves)
        else:
            next_move = best_move
        b.push(next_move)
        if b.is_game_over():
            if b.outcome().winner == True:
                ret_t[i] += 10
            elif b.outcome().winner == False:
                ret_t[i] -= 10
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
    optimizer.step()
    if epoch % 20 == 0:
        L_target = copy.deepcopy(L_train)
        torch.save(L_train, "L")
    epsilon = epsilon ** 1.001
    print("\n")
 
