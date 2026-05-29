import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    # 伯努利多臂⽼⻁机,输⼊K表示拉杆个数
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机⽣成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最⼤的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最⼤的获奖概率
        self.K = K

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该⽼⻁机的k号拉杆获得奖励的概率返回1(获奖)或0(未获奖)
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver:
    # 多臂⽼⻁机算法框架
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护⼀个列表, 记录每⼀步的动作
        self.regrets = []  # 维护⼀个列表, 记录每⼀步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存, k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪⼀根拉杆, 由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运⾏⼀定次数, num_steps为总运⾏次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            # 随机选择⼀根拉杆
            k = np.random.randint(0, self.bandit.K)
        else:
            # 选择期望奖励估值最⼤的拉杆
            k = np.argmax(self.estimates)
            # 得到本次动作的奖励
            r = self.bandit.step(k)
            # 按照参数更新的递推公式进⾏计算更新
            self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers, solver_names):
    # solvers是⼀个列表, 列表中的每个元素是⼀种特定的策略
    # solver_names也是⼀个列表, 存储每个策略的名称
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


np.random.seed(1)  # 设定随机种⼦,使实验具有可重复性
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机⽣成了⼀个%d臂伯努利⽼⻁机" % K)
print("获奖概率最⼤的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_idx,
                                                    bandit_10_arm.best_prob))

np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [
    EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
