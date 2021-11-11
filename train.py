import chess
import torch
import random
import copy
from LOGAN_module import *
from tensorize_board import *

alpha = 0.5
gamma = 0.9
epsilon = 0.1
learning_rate = 0.01
episodes = 2000
epochs = 100
episode_length = 300
L_train = torch.load("L")
L_target = copy.deepcopy(L_train)
#L_train = LOGAN_module()
#L_target = LOGAN_module()

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
        ret_t[i] = torch.tensor((1-alpha)*L_target(ret_x[i].reshape([1,1,8,8])) + alpha * (calculate_material(b, ret_x[i]) +  gamma * -1 * best_move_evaluation))
        r = random.random()
        next_move = moves[0]
        if r < epsilon or not best_move:
            next_move = random.choice(moves)
        else:
            next_move = best_move
        b.push(next_move)
        if b.is_game_over():
            print(b.outcome())
            print(b)
            print(calculate_material(b, ret_x[i]))
            print("game ended after {} moves".format(episode_length))
            return ret_x[:i], ret_t[:i]
    print(b.outcome())
    print(b)
    print(calculate_material(b, ret_x[i]))
    print("game ended after {} moves".format(episode_length))
    print("game ended after {} moves".format(episode_length))
    return ret_x, ret_t

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(L_train.parameters(), lr = learning_rate)
loss = torch.tensor(51)
for episode in range(episodes):
    train_x, train_t = generate_training_episode(gamma, epsilon)
    dataset = torch.utils.data.TensorDataset(train_x, train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 20, shuffle = True)
    for epoch in range(epochs):
        print("episode " + str(episode) +  " epoch: " + str(epoch))
        for data in loader:
            optimizer.zero_grad()
            inputs, labels = data
            outputs = L_train(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(L_train.parameters(), 1.0)
            print("loss: " + str(loss.item()))
            optimizer.step()
        if epoch % 10 == 0:
            L_target = copy.deepcopy(L_train)
            torch.save(L_train, "L")
        epsilon = epsilon * 0.999
        print("\n")
 
