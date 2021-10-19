import chess
import torch

def tensorize_board(board):
    ret = torch.Tensor(67)
    if not board.turn == chess.WHITE:
        board = board.mirror()
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            piece_code = piece.piece_type
            sign = 2*piece.color- 1 #1 for white, -1 for black
            ret[i] = piece_code * sign
        else:
            ret[i] = 0
    if board.has_legal_en_passant():
        ret[-3] = 1
    else:
        ret[-3] = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        ret[-2] = 1
    else:
        ret[-2] = 0
    if board.has_queenside_castling_rights(chess.WHITE):
        ret[-1] = 1
    else:
        ret[-1] = 0
    return ret


if __name__ == "__main__":
    a = chess.Board()
    print(tensorize_board(a))
