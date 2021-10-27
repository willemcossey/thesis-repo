from helper.SimulationJob import SimulationJob
from helper.Distribution import Normal
from math import sqrt
import pandas as pd
import plotly.express as px

# figure 6.2

lamb = 0.003
gamma = 0.1
theta_std = sqrt(gamma * lamb)
print(theta_std)

w_ = [-0.3, 0.7]
rho = 0.5

exponent = 2

sim = SimulationJob(
    gamma,
    Normal(0, theta_std),
    lambda x: int(
        (abs(x - w_[0]) <= rho) | (abs(x - w_[1]) <= rho)
    ),  # Indicator function. Explanation p. 230 (bounded confidence)
    lambda x: (1 - x ** 2) ** exponent,  # P&T p. 248
    1000,
    1000
)

sim.run()

result_df = pd.Series(sim.result, name="opinion")

fig = px.histogram(result_df, x="opinion", nbins=200, histnorm="density")
fig.show()
