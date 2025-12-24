def print_board(board):
    """盤面を表示する関数"""
    print("\n")
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---+---+---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---+---+---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")
    print("\n")

def check_win(board, player):
    """勝利判定を行う関数"""
    # 勝利パターンの定義（横、縦、斜め）
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 横
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 縦
        [0, 4, 8], [2, 4, 6]              # 斜め
    ]
    
    for condition in win_conditions:
        if board[condition[0]] == player and \
           board[condition[1]] == player and \
           board[condition[2]] == player:
            return True
    return False

def check_draw(board):
    """引き分け判定を行う関数"""
    return " " not in board

def main():
    """メインゲームループ"""
    board = [" " for _ in range(9)] # 9つの空マスを作成
    current_player = "〇" # 先攻は〇
    
    print("=== 〇×ゲームへようこそ！ ===")
    print("1〜9の数字を入力してマスを選んでください。\n")
    print(" 1 | 2 | 3 ")
    print("---+---+---")
    print(" 4 | 5 | 6 ")
    print("---+---+---")
    print(" 7 | 8 | 9 ")
    print("===========\n")

    while True:
        print_board(board)
        
        # 入力の受け付け
        try:
            move = int(input(f"{current_player} の番です (1-9): ")) - 1
        except ValueError:
            print("エラー: 数字を入力してください。")
            continue

        # 入力の正当性チェック
        if move < 0 or move > 8:
            print("エラー: 1から9までの数字を選んでください。")
            continue
        if board[move] != " ":
            print("エラー: そのマスは既に埋まっています。")
            continue

        # 盤面の更新
        board[move] = current_player

        # 勝利判定
        if check_win(board, current_player):
            print_board(board)
            print(f"おめでとうございます！ {current_player} の勝ちです！")
            break

        # 引き分け判定
        if check_draw(board):
            print_board(board)
            print("引き分けです！")
            break

        # プレイヤーの交代
        current_player = "×" if current_player == "〇" else "〇"

if __name__ == "__main__":
    main()
