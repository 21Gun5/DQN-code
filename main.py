from maze_env import Maze   # 环境模块，只看思想，未深究代码实现
from RL_brain import DeepQNetwork   # RL的大脑，负责思维和决策

def run_maze():
    step = 0    # 用来控制何时学习的变量；步数，开始为0
    for episode in range(300):  # 共训练300回合/轮
        observation = env.reset()   # 初始化环境，返回state的观测值（state与observation同义），此处为格子位置

        while True:
            env.render()    # 刷新可视化环境（为了能看清每一步是怎么走的）

            action = RL.choose_action(observation)  # 根据观测值即位置来选择动作
            observation_, reward, done = env.step(action)   # 执行动作并返回新观测值（即新格子的地址）、奖励与done标记
            RL.store_transition(observation, action, reward, observation_)      # 将记忆（新旧观测值、动作、奖励）存储下来
            
            # 控制学习的起始时间、频率
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            observation = observation_      # 将新观测值作为下一次的初始观测值（本回合内的下一次）

            # 当天堂/地狱，结束循环，即结束本回合，进入下一个
            if done:
                break

            step += 1       # 总步数+1

    # 全部回合结束，销毁环境
    print('game over')
    env.destroy()


if __name__ == "__main__":
    '''
    同Q-learning，仍是迷宫游戏
    不同的是采取DQN算法
    '''
    env = Maze()    # 构建环境
    # 构建RL机器人，一系列操作（如执行、学习）的执行者
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                    #   output_graph=True
                      )
    env.after(100, run_maze)        # tkinter的形式来调用run_maze函数，100为间隔时间，单位ms
    env.mainloop()      # 环境模块由tkinter实现，在此启动tkinter，即以窗口形式显示环境                 
    RL.plot_cost()      # 显示神经网络的误差曲线