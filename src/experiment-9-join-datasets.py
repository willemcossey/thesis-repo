from helper.Dataset import Dataset


D = Dataset(
    "random",
    {"lmb": [0, 12], "m": [-1, 1]},
    dict(
        gamma=0.005,
        theta_bound="lambda g, w: (1 - g) / (1 + abs(w))",
        p="lambda w: 1",
        d="lambda w: (1 - w ** 2)",
        t_end=10,
        n_samples=10,
        uniform_theta=True,
        lmb_bound=(1 / (3 * 0.005) - 2 / 3 + 0.005 / 3),
        seed=None,
    ),
    children=[
        "7ef9046618b91a136c0019c93f06259b.json",
        "7fbba6e225253163220c0f9045d97481.json",
        "c553d3354f86f9b09faf55e45ee7342b.json",
    ],
)

print(D.name)
print(D.meta["size"])
