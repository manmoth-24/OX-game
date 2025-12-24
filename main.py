import random

class TicTacToe:
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.winner = None

    def reset(self):
        self.board = [" " for _ in range(9)]
        self.winner = None
        return self.get_state()

    def get_state(self):
        return str(self.board) # 盤面を文字列にして状態として扱う

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == " "]

    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            if self.check_win(player):
                self.winner = player
                return True # 勝負あり
            if " " not in self.board:
                self.winner = "Draw"
                return True # 引き分け
            return False # 続行
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
    def __init__(self, player_symbol, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.player = player_symbol
        self.q_table = {} # 経験を記憶する辞書 {(状態, 行動): 価値}
        self.epsilon = epsilon # 探索率（ランダムに動く確率）
        self.alpha = alpha     # 学習率
        self.gamma = gamma     # 割引率

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_moves, training=True):
        # 学習中はたまにランダムに動いて新しい手を試す（探索）
        if training and random.random() < self.epsilon:
            return random.choice(available_moves)
        
        # それ以外は、過去の経験から一番良い手を選ぶ（活用）
        q_values = [self.get_q_value(state, action) for action in available_moves]
        max_q = max(q_values) if q_values else 0
        # 最大値を持つ行動が複数ある場合はランダムに選ぶ
        best_actions = [action for action, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_available_moves):
        """Q値の更新（学習の核心部分）"""
        old_q = self.get_q_value(state, action)
        
        # 次の状態での最大Q値を取得
        if next_available_moves:
            future_max_q = max([self.get_q_value(next_state, a) for a in next_available_moves])
        else:
            future_max_q = 0
        
        # Q学習の更新式
        new_q = old_q + self.alpha * (reward + self.gamma * future_max_q - old_q)
        self.q_table[(state, action)] = new_q

# --- メイン処理 ---

def train_agent(episodes=10000):
    print(f"AIが {episodes} 回の修行（学習）を開始します...")
    game = TicTacToe()
    agent = QLearningAgent("×", epsilon=0.1) # AIは後攻（×）で学習

    for _ in range(episodes):
        state = game.reset()
        done = False
        
        # AI vs ランダムプレイヤーで学習させる
        while not done:
            # --- 先攻（ランダムな相手） ---
            moves = game.available_moves()
            if not moves: break
            p1_move = random.choice(moves)
            done = game.make_move(p1_move, "〇")
            
            if done: # AIが負けた場合
                if game.winner == "〇":
                    # 直前のAIの手に「罰」を与える
                    if last_ai_state is not None:
                        agent.learn(last_ai_state, last_ai_action, -1, state, [])
                break

            # --- 後攻（AI） ---
            state = game.get_state() # 現在の盤面
            moves = game.available_moves()
            action = agent.choose_action(state, moves, training=True)
            
            last_ai_state = state
            last_ai_action = action
            
            done = game.make_move(action, "×")
            next_state = game.get_state()
            next_moves = game.available_moves()
            
            reward = 0
            if done:
                if game.winner == "×": reward = 1   # AI勝ち
                elif game.winner == "Draw": reward = 0.5 # 引き分け
            
            # 学習実行
            agent.learn(state, action, reward, next_state, next_moves)
            
    print("学習完了！最強のAI（後攻）と対戦してください。")
    return agent

def play_vs_ai(agent):
    game = TicTacToe()
    print("\n=== 人間(〇) vs 学習済みAI(×) ===")
    game.print_board()

    while True:
        # --- 人間のターン ---
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
                if game.winner == "〇": print("あなたの勝ちです！（すごい！）")
                else: print("引き分けです。")
                break
                
        except ValueError:
            print("数字を入力してください")
            continue

        # --- AIのターン ---
        print("AIが思考中...")
        state = game.get_state()
        moves = game.available_moves()
        # 学習モードOFF（本気モード）で行動選択
        ai_move = agent.choose_action(state, moves, training=False)
        
        done = game.make_move(ai_move, "×")
        game.print_board()
        
        if done:
            if game.winner == "×": print("AIの勝ちです！")
            else: print("引き分けです。")
            break

if __name__ == "__main__":
    trained_agent = train_agent(10000) # 1万回学習
    play_vs_ai(trained_agent)