import tkinter as tk
from tkinter import messagebox
import random
import pickle
import os

# --- 1. ゲームロジック ---
class TicTacToeLogic:
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

# --- 2. AIエージェント (共通) ---
class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
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
        # まだ経験がない、または全て同じ値ならランダム
        if not q_values: return random.choice(available_moves)
        
        max_q = max(q_values)
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

# --- 3. 両方の手番を覚えるための学習処理 ---
def train_agent_both_sides(filename="tictactoe_brain_v2.pkl"):
    # すでに学習済みファイルがあれば読み込む
    agent = QLearningAgent()
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                agent.q_table = pickle.load(f)
            print(f"学習データ({filename})を読み込みました。")
            return agent
        except:
            print("読み込みエラー。再学習します。")

    print("先攻・後攻の両方を学習中... (約20,000回戦)")
    
    # 共通の脳(Q-table)を持つ2つのエージェントを作成
    # 片方が学習すれば、もう片方も賢くなる（同じ辞書を参照させる）
    agent_p1 = QLearningAgent() # 先攻用
    agent_p2 = QLearningAgent() # 後攻用
    # Qテーブルを共有させる（これが重要！）
    agent_p2.q_table = agent_p1.q_table

    game = TicTacToeLogic()
    episodes = 20000

    for _ in range(episodes):
        state = game.reset()
        done = False
        
        # 状態保持用
        p1_state, p1_action = None, None
        p2_state, p2_action = None, None

        turn = 0 # 偶数=先攻, 奇数=後攻
        while not done:
            moves = game.available_moves()
            current_player_symbol = "〇" if turn % 2 == 0 else "×"
            
            if current_player_symbol == "〇":
                # 先攻(P1)の行動
                action = agent_p1.choose_action(state, moves, training=True)
                p1_state, p1_action = state, action
            else:
                # 後攻(P2)の行動
                action = agent_p2.choose_action(state, moves, training=True)
                p2_state, p2_action = state, action

            # 盤面更新
            done = game.make_move(action, current_player_symbol)
            next_state = game.get_state()
            next_moves = game.available_moves()
            
            # 報酬計算と学習
            if done:
                if game.winner == "〇":
                    # P1勝ち: P1に報酬, P2に罰
                    agent_p1.learn(p1_state, p1_action, 1, next_state, [])
                    if p2_state: agent_p2.learn(p2_state, p2_action, -1, next_state, [])
                elif game.winner == "×":
                    # P2勝ち: P2に報酬, P1に罰
                    if p2_state: agent_p2.learn(p2_state, p2_action, 1, next_state, [])
                    agent_p1.learn(p1_state, p1_action, -1, next_state, [])
                else:
                    # 引き分け: 両方に少し報酬（または0）
                    agent_p1.learn(p1_state, p1_action, 0.5, next_state, [])
                    if p2_state: agent_p2.learn(p2_state, p2_action, 0.5, next_state, [])
            else:
                # 勝負がついていない場合、相手の手番が終わった時点で学習更新する（Q学習の標準的な実装と少し異なるが簡易版として）
                # 今回は厳密なSARSA/Q-Learningの更新タイミング調整より、
                # 「自分の前の手」に対して更新を行う方式をとります。
                if current_player_symbol == "〇":
                    # P2が直前に打っていた場合、P2のQ値を更新（今の盤面はP2にとっての結果）
                    if p2_state:
                        # 報酬0で未来価値を更新
                        agent_p2.learn(p2_state, p2_action, 0, next_state, next_moves)
                else:
                    # P1が直前に打っていた場合
                    if p1_state:
                         agent_p1.learn(p1_state, p1_action, 0, next_state, next_moves)
            
            state = next_state
            turn += 1

    # 学習終了後保存
    with open(filename, "wb") as f:
        pickle.dump(agent_p1.q_table, f)
    print("学習完了！")
    return agent_p1

# --- 4. GUIの実装 ---

class TicTacToeGUI:
    def __init__(self, root, agent):
        self.root = root
        self.root.title("最強AI 〇×ゲーム (先攻後攻えらべる版)")
        self.agent = agent
        self.game = TicTacToeLogic()
        self.buttons = []
        self.game_over = False
        
        # プレイヤー設定 (デフォルトは人間が先攻)
        self.human_side = "〇"
        self.ai_side = "×"
        
        self.create_widgets()
        self.reset_game() # 初期化

    def create_widgets(self):
        # コントロールパネル（上部）
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)
        
        # 先攻後攻選択のラジオボタン
        self.order_var = tk.IntVar(value=1) # 1=人間先攻, 2=人間後攻
        rb1 = tk.Radiobutton(control_frame, text="人間が先攻 (〇)", variable=self.order_var, value=1, command=self.reset_game)
        rb2 = tk.Radiobutton(control_frame, text="人間が後攻 (×)", variable=self.order_var, value=2, command=self.reset_game)
        rb1.pack(side=tk.LEFT, padx=10)
        rb2.pack(side=tk.LEFT, padx=10)

        # ステータス表示
        self.status_label = tk.Label(self.root, text="", font=('Arial', 14))
        self.status_label.pack(pady=5)

        # 盤面フレーム
        board_frame = tk.Frame(self.root)
        board_frame.pack()

        # 3x3のボタン作成
        for i in range(9):
            btn = tk.Button(board_frame, text=" ", font=('Arial', 24), width=4, height=2,
                            command=lambda idx=i: self.on_button_click(idx))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        # リセットボタン
        reset_btn = tk.Button(self.root, text="ゲームリセット", command=self.reset_game, bg="#dddddd")
        reset_btn.pack(pady=10)

    def reset_game(self):
        """ゲームを初期化し、設定に基づいて開始する"""
        self.game.reset()
        self.game_over = False
        
        # ボタンの表示リセット
        for btn in self.buttons:
            btn.config(text=" ", bg="SystemButtonFace")
        
        # 先攻後攻の設定読み込み
        choice = self.order_var.get()
        if choice == 1:
            self.human_side = "〇"
            self.ai_side = "×"
            self.status_label.config(text="あなたの番です (〇)", fg="black")
        else:
            self.human_side = "×"
            self.ai_side = "〇"
            self.status_label.config(text="AIの番です...", fg="black")
            # 人間が後攻なら、AIがいきなり動き出す
            self.root.after(500, self.ai_move)

    def on_button_click(self, index):
        # ゲーム終了済み、埋まってるマス、あるいはAIのターン中はクリック無効
        if self.game_over or self.game.board[index] != " ":
            return
        
        # 人間のターン処理
        self.buttons[index].config(text=self.human_side, fg="blue")
        done = self.game.make_move(index, self.human_side)

        if done:
            self.end_game()
        else:
            # AIのターンへ
            self.status_label.config(text="AIが思考中...")
            self.root.after(500, self.ai_move)

    def ai_move(self):
        if self.game_over: return

        state = self.game.get_state()
        moves = self.game.available_moves()
        
        # AIの手を選ぶ（学習モードOFF）
        action = self.agent.choose_action(state, moves, training=False)
        
        self.buttons[action].config(text=self.ai_side, fg="red")
        done = self.game.make_move(action, self.ai_side)
        
        if done:
            self.end_game()
        else:
            self.status_label.config(text=f"あなたの番です ({self.human_side})")

    def end_game(self):
        self.game_over = True
        winner = self.game.winner
        
        if winner == self.human_side:
            msg = "あなたの勝ちです！"
            color = "blue"
        elif winner == self.ai_side:
            msg = "AIの勝ちです！"
            color = "red"
        else:
            msg = "引き分けです。"
            color = "black"
        
        self.status_label.config(text=msg, fg=color)
        messagebox.showinfo("勝負あり", msg)

# --- メイン実行 ---
if __name__ == "__main__":
    # AIの準備（両方のサイドを学習させる）
    trained_agent = train_agent_both_sides("tictactoe_brain_v2.pkl")

    # ウィンドウの作成
    root = tk.Tk()
    app = TicTacToeGUI(root, trained_agent)
    root.mainloop()