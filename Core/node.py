import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
from Utils.NODE_utils import init_weights, apply_normalization

activations = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
}

def MLP(in_dim, out_dim, hidden_dim, num_layers, non_linearity):
    """Helper function to build an MLP."""
    layers = []
    if num_layers <= 0:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(non_linearity)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(non_linearity)
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """Continuous-time dynamics for the Neural ODE: f(t, x) = dx/dt."""
    
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2,
                 center_logits=False, time_fourier_order=0,
                 time_polynomial_degree=1):
        super(Dynamics, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.non_linearity = activations[non_linearity]
        self.center_logits = center_logits
        if (not isinstance(time_fourier_order, int) or
                isinstance(time_fourier_order, bool) or time_fourier_order < 0):
            raise ValueError("time_fourier_order must be a non-negative integer")
        if (not isinstance(time_polynomial_degree, int) or
                isinstance(time_polynomial_degree, bool) or
                time_polynomial_degree < 1):
            raise ValueError("time_polynomial_degree must be a positive integer")
        self.time_fourier_order = time_fourier_order
        self.time_polynomial_degree = time_polynomial_degree
        self.register_buffer(
            "time_frequencies",
            torch.arange(1, time_fourier_order + 1, dtype=torch.float32),
        )
        
        # Polynomial degree 1 and Fourier order 0 recover scalar-time input.
        time_feature_dim = time_polynomial_degree + 2 * time_fourier_order
        self.net = MLP(
            input_dim + time_feature_dim,
            input_dim, hidden_dim, N_layers, self.non_linearity
        )

    def time_features(self, t, batch_size, *, dtype, device):
        """Return [t,...,t^p,sin(2πt),cos(2πt),...,sin(2πmt),cos(2πmt)]."""
        scalar = t.to(dtype=dtype, device=device).reshape(1, 1).expand(batch_size, 1)
        powers = torch.arange(
            1, self.time_polynomial_degree + 1, dtype=dtype, device=device
        ).reshape(1, -1)
        polynomial = scalar ** powers
        if self.time_fourier_order == 0:
            return polynomial
        frequencies = self.time_frequencies.to(dtype=dtype, device=device).reshape(1, -1)
        phase = 2.0 * math.pi * scalar * frequencies
        periodic = torch.stack(
            [torch.sin(phase), torch.cos(phase)], dim=-1
        ).reshape(batch_size, -1)
        return torch.cat([polynomial, periodic], dim=1)

    def forward(self, t, x):
        """Compute the dynamics f(t, x) = dx/dt."""
        batch_size = x.shape[0]
        t_tensor = self.time_features(
            t, batch_size, dtype=x.dtype, device=x.device
        )
        state = x - x.mean(dim=-1, keepdim=True) if self.center_logits else x
        x_input = torch.cat([state, t_tensor], dim=1)
        
        dx = self.net(x_input)
        
        if self.center_logits:
            dx = dx - dx.mean(dim=-1, keepdim=True)
        return dx


class Semiflow(nn.Module):
    """Solves the Neural ODE with configurable normalization."""
    
    def __init__(self, dynamics, T=1.0, step_size=0.01, method='rk4', 
                 normalization='softmax', softmax_epsilon=1e-8,
                 solver_rtol=1e-7, solver_atol=1e-9):
        super(Semiflow, self).__init__()
        self.dynamics = dynamics
        self.T = T
        self.dt = step_size
        self.method = method
        self.normalization = normalization
        self.softmax_epsilon = softmax_epsilon
        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol

    def weights_to_state(self, x):
        """Map simplex weights to the latent coordinates integrated by the ODE."""
        if not torch.isfinite(x).all() or torch.any(x < 0):
            raise ValueError("Initial mixture weights must be finite and non-negative")
        if torch.any(x.sum(dim=-1, keepdim=True) <= 0):
            raise ValueError("Initial mixture weights must have positive total mass")
        if self.normalization == "softmax":
            alpha0_eps = x + self.softmax_epsilon
            alpha0_eps = alpha0_eps / alpha0_eps.sum(dim=-1, keepdim=True)
            state0 = torch.log(alpha0_eps)
        elif self.normalization == "softplus":
            # Stable inverse softplus: starts (approximately) at the observed
            # mixture weights while evolving unconstrained coordinates.
            positive_x = x.clamp_min(self.softmax_epsilon)
            state0 = positive_x + torch.log(-torch.expm1(-positive_x))
        else:
            state0 = x
        return state0

    def state_to_weights(self, state):
        """Map latent NODE coordinates to ordered simplex weights."""
        if self.normalization == "softmax":
            return torch.softmax(state, dim=-1)
        return apply_normalization(
            state, self.normalization, eps=self.softmax_epsilon
        )

    def integrate_state(self, state0, eval_times=None):
        """Integrate an already parametrized latent initial state."""
        if eval_times is None:
            integration_time = torch.tensor([0.0, self.T]).to(state0)
        else:
            integration_time = eval_times.to(state0)

        if self.method == 'dopri5' or self.method == 'adams':
            out = odeint(
                self.dynamics, state0, integration_time,
                method=self.method, rtol=self.solver_rtol, atol=self.solver_atol
            )
        else:
            out = odeint(
                self.dynamics, state0, integration_time,
                method=self.method, rtol=self.solver_rtol, atol=self.solver_atol,
                options={'step_size': self.dt}
            )

        out = self.state_to_weights(out)
        return out if eval_times is not None else out[1]

    def forward(self, x, eval_times=None):
        """
        Solve the ODE from fixed initial mixture weights.

        ``softmax`` evolves logits; ``l1`` preserves the original
        ReLU-plus-L1 normalization; ``softplus`` is its smooth counterpart.
        """
        return self.integrate_state(self.weights_to_state(x), eval_times)


class NeuralODE(nn.Module):
    """Neural ODE with configurable normalization method."""
    
    def __init__(self, input_dim, hidden_dim, non_linearity='relu', N_layers=2,
                 T=1.0, step_size=0.01, method='rk4', normalization='softmax',
                 softmax_epsilon=1e-8, solver_rtol=1e-7, solver_atol=1e-9,
                 time_fourier_order=0, time_polynomial_degree=1):
        super(NeuralODE, self).__init__()
        self.input_dim = input_dim
        self.T = T
        self.dt = step_size
        self.method = method
        self.activation_type = non_linearity
        self.normalization = normalization
        self.time_fourier_order = time_fourier_order
        self.time_polynomial_degree = time_polynomial_degree
        if normalization not in {"softmax", "l1", "softplus"}:
            raise ValueError(
                "normalization must be 'softmax', 'l1' or 'softplus'"
            )

        self.dynamics = Dynamics(
            input_dim, hidden_dim, non_linearity, N_layers,
            center_logits=(normalization == "softmax"),
            time_fourier_order=time_fourier_order,
            time_polynomial_degree=time_polynomial_degree,
        )
        
        self.flow = Semiflow(
            self.dynamics, T, step_size, method, normalization,
            softmax_epsilon=softmax_epsilon,
            solver_rtol=solver_rtol,
            solver_atol=solver_atol
        )
        
        self.apply(lambda m: init_weights(m, self.activation_type))

        # The initial state is configured separately because it depends on
        # the first fitted mixture weight vector for each participant.
        self.register_buffer("initial_state_base", None)
        self.register_parameter("initial_state_delta", None)

    def configure_initial_state(self, initial_weights, trainable=True):
        """Initialize z(0) from observed weights and optionally learn it."""
        state = self.flow.weights_to_state(initial_weights).detach().clone()
        if self.normalization == "softmax":
            # Remove the additive softmax gauge so z(0) is identifiable.
            state = state - state.mean(dim=-1, keepdim=True)
        self.initial_state_base = state
        delta = torch.zeros_like(state)
        if trainable:
            self.initial_state_delta = nn.Parameter(delta)
        else:
            self.initial_state_delta = None
        return self

    @property
    def has_trainable_initial_state(self):
        return isinstance(self.initial_state_delta, nn.Parameter)

    def initial_latent_state(self):
        """Return the fitted latent z(0), with the softmax gauge fixed."""
        if self.initial_state_base is None:
            raise RuntimeError("The trainable initial state has not been configured")
        state = self.initial_state_base
        if self.initial_state_delta is not None:
            state = state + self.initial_state_delta
        if self.normalization == "softmax":
            state = state - state.mean(dim=-1, keepdim=True)
        return state

    def fitted_initial_weights(self):
        """Return the simplex weights implied by the current fitted z(0)."""
        return self.flow.state_to_weights(self.initial_latent_state())
    
    def forward(self, x=None, eval_times=None):
        if self.initial_state_base is not None:
            return self.flow.integrate_state(self.initial_latent_state(), eval_times)
        if x is None:
            raise ValueError("Fixed-initial NODE requires initial mixture weights")
        return self.flow(x, eval_times)
