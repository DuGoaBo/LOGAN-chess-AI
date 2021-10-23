import chess
import torch
piece_values = {1: 1., 2: 3., 3: 3.4, 4: 5, 5: 9, 6: 200.}

def tensorize_board(board):
    ret = torch.Tensor(1, 1, 8, 8)
    if not board.turn == chess.WHITE:
        board = board.mirror()
    for i in range(64):
        x = (64 - i - 1) // 8
        y = i % 8
        piece = board.piece_at(i)
        if piece is not None:
            piece_value = piece_values[piece.piece_type]
            sign = 2*piece.color- 1 #1 for white, -1 for black
            ret[0,0,x,y] = piece_value * sign
        else:
            ret[0,0,x,y] = 0
    if board.has_legal_en_passant():
        i = board.ep_square
        x = (64 - i - 1) // 8
        y = i%8
        ret[0,0,x,y] += 0.5
    if board.has_kingside_castling_rights(chess.WHITE):
        ret[0,0,7,4] += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        ret[0,0,7,4] += 1
    return ret

def calculate_material(board, tensorized_board):
    if board.is_game_over():
        if board.outcome().winner == chess.WHITE:
            return 200
        elif board.outcome().winner == chess.BLACK:
            return -200
    ret = tensorized_board.sum().item()
    if board.has_kingside_castling_rights(chess.WHITE):
        ret -= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        ret -= 1
    if board.has_legal_en_passant():
        ret -= 0.5
    return ret

if __name__ == "__main__":
    a = chess.Board()
    a.push_san('e4')
    a.push_san('e6')
    a.push_san('e5')
    a.push_san('d5')
    print(tensorize_board(a).float())
    print(calculate_material(a, tensorize_board(a)))
