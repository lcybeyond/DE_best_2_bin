import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Population:
    def __init__(self, min_range, max_range, dim, factor, rounds, size, object_func, CR=0.75):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.CR = CR
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range) for s in range(self.dimension)])
                              for tmp in range(self.size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None
        self.fig = plt.figure()

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            # 保证r0,r1,r2,i互不相同
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i or r1 == i or r2 == i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            # 变异
            tmp = self.individuality[r0] + (self.individuality[r1] + self.individuality[r2]) * self.factor
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)

    def crossover_and_select(self):
        for i in range(self.size):
            # Jrand保证有一个dimension是变异的
            Jrand = random.randint(0, self.dimension)
            # 交叉
            for j in range(self.dimension):
                if random.random() > self.CR and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
            # 计算值
            tmp = self.get_object_function_value(self.mutant[i])
            # 选择
            if tmp < self.object_function_values[i]:
                self.individuality[i] = self.mutant[i]
                self.object_function_values[i] = tmp

    def print_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        print('轮数:' + str(self.cur_round))
        print('最佳个体:' + str(self.individuality[i]))
        print('目标函数值:' + str(m))

    def plot_3d(self):
        ax = Axes3D(self.fig)
        plt.ion()
        X = np.linspace(self.min_range, self.max_range, 100)
        Y = np.linspace(self.min_range, self.max_range, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.get_object_function_value([X, Y])
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for i in range(self.size):
            x = self.individuality[i][0]
            y = self.individuality[i][1]
            z = self.get_object_function_value([x, y])
            ax.scatter(x, y, z, c='black', marker='o')
        plt.show()
        plt.pause(0.01)

    def evolution(self):
        while self.cur_round < self.rounds:
            self.plot_3d()
            self.mutate()
            self.crossover_and_select()
            self.print_best()
            self.cur_round = self.cur_round + 1


if __name__ == "__main__":
    def f(v):
        return 100.0 * (v[1] - v[0] ** 2.0) ** 2.0 + (1 - v[0]) ** 2.0


    p = Population(min_range=-2.048, max_range=2.048, dim=2, factor=0.8, rounds=100, size=100, object_func=f)
    p.evolution()
