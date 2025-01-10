import numpy as np
import random

# 環境の設定
class RobotEnvironment:
    def __init__(self, g_p, s_p):
        self.g_p = g_p  #goal position
        self.s_p = s_p  # 状態の範囲（数直線上の位置）
        self.reset()

    def reset(self):
        self.position = random.choice(self.s_p)  # 初期位置をランダムに設定
        return self.position

    def step(self, action):
        # action: 0 -> 左に1m, 1 -> 右に1m
        if action == 0:
            self.position -= 1
        elif action == 1:
            self.position += 1
        
        # 範囲外を防ぐ
        self.position = max(min(self.position, max(self.s_p)), min(self.s_p))
        
        # 報酬と終了判定
        if self.position == self.g_p:
            reward = 1  # ゴールに到達
            flag = True
        else:
            reward = -0.1  # それ以外は小さなペナルティ
            flag = False
        
        return self.position, reward, flag

# Q学習の設定
class QLearningAgent:
    def __init__(self, potision, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((len(potision), len(action_space)))  # Q値を初期化
        self.potision = potision
        self.action_space = action_space
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # ε-greedy法の探索率

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.action_space)  # ランダムな行動を選択（探索）
        else:
            return np.argmax(self.q_table[state])  # 最大のQ値を持つ行動を選択（活用）

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# 実行部分
if __name__ == "__main__":
    # 初期化
    number_potision = list(range(-10, 11))  # -10から10までの数直線
    action_space = [0, 1]  # 左に1m（0）または右に1m（1）
    goal = 0  # ゴールの位置を0に設定
    env = RobotEnvironment(goal, number_potision)
    agent = QLearningAgent(number_potision, action_space)

    # 学習プロセス
    episodes = 1000
    total_rewards = []  # 各エピソードの報酬の合計を記録

    for episode in range(episodes):
        state = env.reset()
        Flag = False
        episode_reward = 0  # エピソードごとの累積報酬

        while not Flag:
            action = agent.choose_action(number_potision.index(state))
            next_state, reward, Flag = env.step(action)
            agent.update_q_table(
                number_potision.index(state), action, reward, number_potision.index(next_state)
            )
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    # 学習結果の確認
    print("\n学習後のQ値テーブル:")
    for i, q_values in enumerate(agent.q_table):
        print(f"状態 {number_potision[i]}: {q_values}")

    print(f"\nトータルの報酬: {sum(total_rewards)}")
    print(f"平均報酬: {np.mean(total_rewards)}")
