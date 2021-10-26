from helper.SimulationJob import SimulationJob
from helper.Distribution import Normal
from math import sqrt

# figure 6.2

lamb = 0.003
gamma = 0.25
std = sqrt(gamma * lamb)

w_ = [-0.3, 0.7]
rho = 0.5

exponent = 0.5

sim = SimulationJob(
    gamma,
    Normal(0, std),
    lambda x: int((abs(x - w_[0]) <= rho) | (abs(x - w_[1]) <= rho)),
    lambda x: (1 - x**2) ** exponent,
    1,
    100,
)

sim.run()

print(sim.result)
