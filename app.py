from flask import Flask, render_template, request, jsonify
import pickle
import random
import os

app = Flask(__name__)

# --- ゲームロジックとAIクラス (前回と同じ) ---

class TicTacToeLogic:
    def __init__(self, board=None):
        self.board = board if board else [" " for _ in range(9)]

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == " "]

    def check_win(self, player):
        win_conditions = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        return any(all(self.board[i] == player for i in wc) for wc in win_conditions)

    def is_draw(self):
        return " " not in self.board and not self.check_win("〇") and not self.check_win("×")

class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def choose_action(self, state, available_moves):
        # Web版は常に本気モード(学習なし)なのでepsilonなどは不要
        q_values = [self.q_table.get((state, action), 0.0) for action in available_moves]
        if not q_values: return random.choice(available_moves)
        max_q = max(q_values)
        best_actions = [action for action, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

# --- AIの準備 ---
agent = QLearningAgent()
model_path = "tictactoe_brain_v2.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        agent.q_table = pickle.load(f)
    print("AIの脳をロードしました。")
else:
    print("警告: 学習データが見つかりません。ランダムに動きます。")

# --- Webルート設定 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/move', methods=['POST'])
def api_move():
    data = request.json
    board = data['board']     # 現在の盤面リスト
    ai_symbol = data['aiSide'] # AIはどっち？ ('〇' or '×')
    
    game = TicTacToeLogic(board)
    
    # すでに勝負がついているかチェック
    if game.check_win("〇") or game.check_win("×") or game.is_draw():
        return jsonify({'move': None, 'game_over': True})

    # AIの手番
    state = str(board)
    moves = game.available_moves()
    
    if not moves:
        return jsonify({'move': None, 'game_over': True})

    # AIが手を決める
    action = agent.choose_action(state, moves)
    
    return jsonify({
        'move': action, 
        'game_over': False
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)


from werkzeug.middleware.proxy_fix import ProxyFix


# 【重要】Renderなどのプロキシ下で動かすための設定
# x_for=1 は "X-Forwarded-For ヘッダーを1階層分信頼する" という意味です
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

@app.route('/api/move', methods=['POST'])
def move():
    # これで自動的に本当のIPアドレスが取れるようになります
    client_ip = request.remote_addr 
    print(f"User IP: {client_ip}")
    return "OK"