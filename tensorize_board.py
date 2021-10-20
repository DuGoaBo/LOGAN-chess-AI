import chess
import torch

def tensorize_board(board):
    ret = torch.Tensor(1, 1, 8, 8)
    if not board.turn == chess.WHITE:
        board = board.mirror()
    for i in range(64):
        piece = board.piece_at(i)
        x = (64 - i - 1) // 8
        y = i % 8
        if piece is not None:
            piece_code = piece.piece_type
            sign = 2*piece.color- 1 #1 for white, -1 for black
            ret[0,0,x,y] = piece_code * sign
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
    a.push_san('exd6')
    print(tensorize_board(a))
    print(calculate_material(a, tensorize_board(a)))
