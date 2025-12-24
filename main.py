import random
import pickle  # データを保存・読み込みするためのライブラリ
import os      # ファイルの存在確認用

class TicTacToe:
    """ゲームのルールを管理するクラス（変更なし）"""
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.winner = None

    def reset(self):
        self.board = [" " for _ in range(9)]
        self.winner = None
        return self.get_state()

    def get_state(self):
        return str(self.board)

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == " "]

    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            if self.check_win(player):
                self.winner = player
                return True
            if " " not in self.board:
                self.winner = "Draw"
                return True
            return False
        return False

    def check_win(self, player):
        win_conditions = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        return any(all(self.board[i] == player for i in wc) for wc in win_conditions)

    def print_board(self):
        b = self.board
        print(f"\n {b[0]} | {b[1]} | {b[2]} ")
        print("---+---+---")
        print(f" {b[3]} | {b[4]} | {b[5]} ")
        print("---+---+---")
        print(f" {b[6]} | {b[7]} | {b[8]} \n")

class QLearningAgent:
    """AIエージェントクラス（保存・読み込み機能を追加）"""
    def __init__(self, player_symbol, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.player = player_symbol
        self.q_table = {} 
        self.epsilon = epsilon 
        self.alpha = alpha     
        self.gamma = gamma     

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(available_moves)
        
        q_values = [self.get_q_value(state, action) for action in available_moves]
        max_q = max(q_values) if q_values else 0
        best_actions = [action for action, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_available_moves):
        old_q = self.get_q_value(state, action)
        if next_available_moves:
            future_max_q = max([self.get_q_value(next_state, a) for a in next_available_moves])
        else:
            future_max_q = 0
        new_q = old_q + self.alpha * (reward + self.gamma * future_max_q - old_q)
        self.q_table[(state, action)] = new_q

    # --- 追加機能: 脳（Qテーブル）の保存 ---
    def save_brain(self, filename="tictactoe_brain.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"学習データを '{filename}' に保存しました。")

    # --- 追加機能: 脳（Qテーブル）の読み込み ---
    def load_brain(self, filename="tictactoe_brain.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"学習データを '{filename}' から読み込みました。")

# --- メイン処理 ---

def train_agent(episodes=10000, filename="tictactoe_brain.pkl"):
    print(f"学習データが見つかりません。{episodes} 回の修行を開始します...")
    game = TicTacToe()
    agent = QLearningAgent("×", epsilon=0.1)

    for _ in range(episodes):
        state = game.reset()
        done = False
        last_ai_state = None
        last_ai_action = None
        
        while not done:
            # 先攻（ランダム）
            moves = game.available_moves()
            if not moves: break
            p1_move = random.choice(moves)
            done = game.make_move(p1_move, "〇")
            
            if done:
                if game.winner == "〇":
                    if last_ai_state is not None:
                        agent.learn(last_ai_state, last_ai_action, -1, state, [])
                break

            # 後攻（AI）
            state = game.get_state()
            moves = game.available_moves()
            action = agent.choose_action(state, moves, training=True)
            last_ai_state = state
            last_ai_action = action
            
            done = game.make_move(action, "×")
            next_state = game.get_state()
            next_moves = game.available_moves()
            
            reward = 0
            if done:
                if game.winner == "×": reward = 1
                elif game.winner == "Draw": reward = 0.5
            
            agent.learn(state, action, reward, next_state, next_moves)
            
    print("学習完了！")
    agent.save_brain(filename) # 学習が終わったら保存
    return agent

def main():
    filename = "tictactoe_brain.pkl"
    agent = QLearningAgent("×") # AI作成

    # ファイルがあるか確認
    if os.path.exists(filename):
        # あれば読み込む
        agent.load_brain(filename)
    else:
        # なければ学習して保存
        agent = train_agent(10000, filename)

    # いざ対戦
    game = TicTacToe()
    print("\n=== 人間(〇) vs 学習済みAI(×) ===")
    game.print_board()

    while True:
        # 人間のターン
        try:
            moves = game.available_moves()
            if not moves: break
            
            human_move = int(input("あなたの番です (1-9): ")) - 1
            if human_move not in moves:
                print("そこには置けません")
                continue
            
            done = game.make_move(human_move, "〇")
            game.print_board()
            
            if done:
                if game.winner == "〇": print("あなたの勝ちです！")
                else: print("引き分けです。")
                break
                
        except ValueError:
            print("数字を入力してください")
            continue

        # AIのターン
        print("AIが思考中...")
        state = game.get_state()
        moves = game.available_moves()
        # 学習モードOFF（本気モード）
        ai_move = agent.choose_action(state, moves, training=False)
        
        done = game.make_move(ai_move, "×")
        game.print_board()
        
        if done:
            if game.winner == "×": print("AIの勝ちです！")
            else: print("引き分けです。")
            break

if __name__ == "__main__":
    main()