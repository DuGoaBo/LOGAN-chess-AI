import chess
import torch

def tensorize_board(board):
    ret = torch.Tensor(8*8 + 3)
    if not board.turn == chess.WHITE:
        board.apply_mirror()
    for i in range(8 * 8):
        piece = board.piece_at(i)
        if piece is not None:
            ret[i] = piece.piece_type
        else:
            ret[i] = 0
    if board.has_legal_en_passant():
        ret[-3] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        ret[-2] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        ret[-1] = 1
    return ret


if __name__ == "__main__":
    a = chess.Board()
    print(tensorize_board(a))
