import numpy as np
import pandas as pd
import tensorflow as tf
from maze_env import Maze

np.random.seed(1)   # 设置随机数生成器的种子；若设置相同的种子，则每次生成的随机数也相同，若不设置则随机数不同
tf.set_random_seed(1)   # python有自带的seed函数，此处np、tf的设置种子函数与其同思想

# 离线学习的DQN算法
class DeepQNetwork:
    # 定义初始值
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False, # 是否输出tensorboard文件
    ):
        self.n_actions = n_actions      # 动作空间的大小，为4
        self.n_features = n_features    # 观测值的维度；为2；如(x,y）坐标表示一个点，就可说state的维度为2，即feature为2；基本思想便是通过feature来预测action（在这feature与state同义）
        self.lr = learning_rate         # 学习率
        self.gamma = reward_decay       # 奖励衰减值
        self.epsilon_max = e_greedy     # 贪婪度（根据Q表选行为的概率，否则其他方式，如随机选）
        self.replace_target_iter = replace_target_iter      # 替换target_net参数的频率（每x步）
        self.memory_size = memory_size      # 记忆上限
        self.batch_size = batch_size        # 神经网络中，随机梯度下降时用到的批量；批量是在单次迭代中，计算梯度的样本总数；
        self.epsilon_increment = e_greedy_increment     #
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max     # 

        self.learn_step_counter = 0     # 总的学习步数

        # 初始化记忆
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))      # 初始值为0；要存[s,a,r,s_]，s和s_各有2个维度，a和r各1个，故x2+2

        # 建立所需的2个神经网络
        self._build_net()
        
        # 替代参数
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 创建session（tensorflow运行必需的）
        self.sess = tf.Session()

        # 是否输出tensorboard文件
        if output_graph:
            # $ tensorboard --logdir=logs，shell中运行，将其作为日志路径；随后IP:6006访问
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())       # 初始化所有变量；tf要用sess的run来运行功能语句，是其基本语法
        self.cost_his = []      # 记录误差，plot_cost会用到

    # 构建神经网络
    def _build_net(self):
        # 构建 evaluate_net 神经网络，及时提升/更新参数
        # 共两个输入：state状态值、Q值
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入，用来接收observation观测值，即状态值
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 输入，用来接收q_target的值，即Q值
        # 添加层
        with tf.variable_scope('eval_net'):     # 层的配置
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]    # c_names是一个集合，里面存储着变量；要通过集合名来调用变量
            n_l1 = 10       # 第一层有10个节点；有上下左右4个动作，故第二层有4个节点）
            w_initializer = tf.random_normal_initializer(0., 0.3)       # 权重weight的初始值，为介于0-0.3的随机值
            b_initializer = tf.constant_initializer(0.1)        # 偏移bias的初始值，为常数0.1

            # 第一层
            with tf.variable_scope('l1'):
                # 用c_names集合中的变量，对权重和偏移进行赋值
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # 计算真实值，与预测值相比较得损失
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)     # 由(s,w1,b1)而得l1层真实值；matmul为矩阵乘法；relu激励函数，将直线“掰弯”变为非线性

            # 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2        # 无激励函数，仍线性
        
        # 构建target_net神经网络，（注释参看eval_net的)
        # 一个输入
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        # 添加层
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # 第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

        # 定义损失如何计算
        with tf.variable_scope('loss'):
            # target为神经网络得到的预测值，eval为真实值，通过二者得损失
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))       # 先二者差的平方，再平均值

        # 训练（通过优化器减少损失)
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)     # lr为学习率；使用优化器，力求减少损失

    # 存储记忆（当前和下一步state、行为、奖励）
    def store_transition(self, s, a, r, s_):
        # 如果对象没有此属性，则创建且初始化为0
        if not hasattr(self, 'memory_counter'):     # hasattr判断对象是否包含对应的属性
            self.memory_counter = 0

        # (s,a,r,s_)为一条记忆记录
        transition = np.hstack((s, [a, r], s_))     # hstack沿着水平方向将数组堆叠起来（vstack沿竖直方向

        # 用新记录代替旧记录
        index = self.memory_counter % self.memory_size      # 计算索引；记忆是有上限的，故满了之后，要覆盖旧的；不论满与不满，索引都可(%size)得
        self.memory[index, :] = transition      # 插入记录或称覆盖记录

        self.memory_counter += 1    # 记忆数量+1

    # 选择行为
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]      # 原本观测值的维数与tf输入所需的不符，故升维

        # 贪婪度=0.x，x0%的可能按最优Q值选择行为
        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})     # 变量赋值，并得到相应动作的Q值
            action = np.argmax(actions_value)       # 选择具有最大值的动作

        # 贪婪度=0.x，1-x0%的可能随机选行为
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 学习
    def learn(self):
        # 按照频率来替代参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从记忆库中抽取批量记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 得到相应的Q值
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # 训练神经网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 显示误差，即学习效果（可选）
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')      # 纵坐标名称
        plt.xlabel('training steps')        # 横坐标名称
        plt.show()      # 显示图像

if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

    observation = env.reset()
    print(observation)


