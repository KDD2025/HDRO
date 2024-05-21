import numpy as np
from scipy.optimize import minimize


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_constraint(x, target_distribution, kl_target):
    return kl_divergence(target_distribution, x) - kl_target

def find_distribution_with_kl(target_distribution, kl_target):
    n = len(target_distribution)

    # 初始猜测一个分布
    initial_guess = np.ones(n) / n

    # 定义KL散度为目标函数
    objective_function = lambda x: np.sum((target_distribution - x) ** 2)  # 这里使用平方和作为目标函数

    # 约束条件：分布的和为1和KL散度为目标值
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: kl_constraint(x, target_distribution, kl_target)})

    # 寻找最小目标函数的分布
    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

    # 输出结果
    return result.x
#

# 示例用法
target_distribution = np.array([210747/(210747+594559+194903), 594559/(210747+594559+194903), 194903/(210747+594559+194903)])
kl_target = 1.0
new_distribution = find_distribution_with_kl(target_distribution, kl_target)
rounded_values = [round(value, 1) for value in new_distribution]
sum_rounded_values = sum(rounded_values)
rounded_values[-1] += 1 - sum_rounded_values
# 调整最后一个元素，使得四舍五入后的元素之和等于1
rounded_values[-1] = round(1 - sum(rounded_values[:-1]), 1)

print("rounded_values",rounded_values)





