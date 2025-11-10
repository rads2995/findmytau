import cupy as cp
import xgboost as xgb

class SDE:
    def __init__(
        self,
        N: cp.int64 = 10000,
        M: cp.int64 = 2000,
        mu: cp.float64 = 0.05,
        sigma: cp.float64 = 0.2,
        x0: cp.float64 = 1.0,
        T: cp.float64 = 1.0
    ):
        self.N: cp.int64 = N
        self.M: cp.int64 = M
        
        self.mu: cp.float64 = mu
        self.sigma: cp.float64 = sigma
        self.x0: cp.float64 = x0

        self.T: cp.float64 = T
        self.dt: cp.float64 = self.T / self.M

        self.X = cp.zeros((self.M + 1, self.N), dtype=cp.float64)
        self.X[0, :] = self.x0

        self.alpha: cp.float64 = 0.5
        self.stopping_times = cp.full(self.N, self.M, dtype=cp.int64)

    def simulate(self):

        rng = cp.random.default_rng()
        for t in range(self.M):
            dW_X = rng.standard_normal(size=self.N, dtype=cp.float64) * cp.sqrt(self.dt)
            self.X[t + 1, :] = self.X[t, :] * cp.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW_X)

    def optimal_stopping(self):
        
        V: cp.float64 = cp.maximum(1.0 - self.X[-1, :], 0.0)
        discount: cp.float64 = cp.exp(-0.05 * self.dt)

        for t in range(self.M - 1, -1, -1):
            item = (1.0 - self.X[t, :]) > 0.0
            
            V = discount * V
            
            if not cp.any(item):
                continue
            
            features = self.X[t, item].reshape(-1, 1)            
            Y = V[item]

            # TODO: pass cupy arrays directly once rocm's xgboost is updated
            dtrain = xgb.QuantileDMatrix(features.get(), Y.get())

            evals = [(dtrain, 'train')]

            model = xgb.train(
                {
                "objective": "reg:quantileerror",
                "tree_method": 'hist',
                "quantile_alpha": self.alpha,
                "learning_rate": 0.04,
                "max_depth": 5,
                "device": "gpu"
                },
                dtrain,
                num_boost_round=64,
                evals=evals,
                early_stopping_rounds=5
            )

            # TODO: inplace predict supports cupy arrays on recent versions
            dfeatures = xgb.DMatrix(features.get())
            C = cp.asarray(model.predict(dfeatures))

            exercise_payoff = cp.maximum(1.0 - self.X[t, item], 0.0)
            exercise = exercise_payoff >= C
            V[item] = cp.where(exercise, exercise_payoff, V[item])

            self.stopping_times[item] = cp.where(
                exercise & (self.stopping_times[item] > t),
                t,
                self.stopping_times[item]
            )

        return cp.mean(V)
