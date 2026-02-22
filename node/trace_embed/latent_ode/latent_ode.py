import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax


# ---------------------------------------------------------------------------
# Model (JAX / Equinox)
# ---------------------------------------------------------------------------

class ODEFunc(eqx.Module):
    """Defines dz/dt = f(z, t). A small MLP operating in latent space."""
    mlp: eqx.nn.MLP

    def __init__(self, d_z: int, d_hidden: int, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=d_z,
            out_size=d_z,
            width_size=d_hidden,
            depth=3,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, _t: float, z: jax.Array, _args) -> jax.Array:
        return self.mlp(z)


class GRUEncoder(eqx.Module):
    """Projects embeddings, runs a backwards GRU, outputs (mu, log_var) for z0."""
    proj: eqx.nn.Linear
    gru: eqx.nn.GRUCell
    mu_head: eqx.nn.Linear
    log_var_head: eqx.nn.Linear

    def __init__(self, d_embed: int, d_proj: int, d_encoder: int, d_z: int, *, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.proj = eqx.nn.Linear(d_embed, d_proj, key=k1)
        self.gru = eqx.nn.GRUCell(d_proj, d_encoder, key=k2)
        self.mu_head = eqx.nn.Linear(d_encoder, d_z, key=k3)
        self.log_var_head = eqx.nn.Linear(d_encoder, d_z, key=k4)

    def __call__(self, x: jax.Array, mask: jax.Array) -> tuple[jax.Array, jax.Array]:
        # x: (T, d_embed), mask: (T,) bool
        h = jax.vmap(self.proj)(x)  # (T, d_proj)

        # Reverse sequence for backwards GRU : 
        # Encoder is trying to produce: z₀, the initial latent state at t=0
        # Padding (mask=False) lands at the start of the reversed sequence, so the
        # GRU hidden state is only updated on valid steps.
        h_rev = jnp.flip(h, axis=0)
        mask_rev = jnp.flip(mask, axis=0)

        def step(carry, x_m):
            xi, mi = x_m
            h_new = self.gru(xi, carry)
            # Keep previous hidden state for padded steps
            h_out = jnp.where(mi, h_new, carry)
            return h_out, None

        h0 = jnp.zeros(self.gru.hidden_size)
        h_final, _ = jax.lax.scan(step, h0, (h_rev, mask_rev))

        mu = self.mu_head(h_final)
        log_var = self.log_var_head(h_final)
        return mu, log_var


class Decoder(eqx.Module):
    """Maps latent states back to embedding space."""
    mlp: eqx.nn.MLP

    def __init__(self, d_z: int, d_embed: int, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=d_z,
            out_size=d_embed,
            width_size=d_z * 2,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.mlp(z)


class LatentODE(eqx.Module):
    encoder: GRUEncoder
    ode_func: ODEFunc
    decoder: Decoder

    def __init__(self, d_embed: int, config, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = GRUEncoder(d_embed, config.d_proj, config.d_encoder, config.d_z, key=k1)
        self.ode_func = ODEFunc(config.d_z, config.d_ode_hidden, key=k2)
        self.decoder = Decoder(config.d_z, d_embed, key=k3)

    def encode(
        self, x: jax.Array, mask: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Encode a single trajectory and sample z0 via reparameterisation."""
        mu, log_var = self.encoder(x, mask)
        # log_var = log(sigma^2), so sigma = exp(0.5 * log_var)
        # reparameterzation trick -> sample from standard normal
        z0 = mu + jnp.exp(0.5 * log_var) * jax.random.normal(key, mu.shape)
        return z0, mu, log_var

    def solve(self, z0: jax.Array, ts: jax.Array) -> jax.Array:
        """Solve the ODE from z0, saving states at each timestamp in ts."""
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode_func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=z0,
            saveat=diffrax.SaveAt(ts=ts),
            # TODO : rtol, atol in config
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-5),
        )
        return solution.ys  # (T_max, d_z)

    def decode(self, zs: jax.Array) -> jax.Array:
        return jax.vmap(self.decoder)(zs)  # (T_max, d_embed)


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------

def elbo_single(
    model: LatentODE,
    x: jax.Array,       # (T_max, d_embed)
    mask: jax.Array,    # (T_max,) bool
    ts: jax.Array,      # (T_max,) float
    key: jax.Array,
    beta: float,
) -> jax.Array:
    # sample z
    z0, mu, log_var = model.encode(x, mask, key)
    # odesolve
    zs = model.solve(z0, ts)        # (T_max, d_z)
    x_hat = model.decode(zs)        # (T_max, d_embed)

    # ELBO recon term
    # MSE reconstruction over valid steps only
    recon = jnp.sum(mask[:, None] * (x_hat - x) ** 2) / jnp.sum(mask)

    # ELBO KL term
    # KL( N(mu, sigma^2) || N(0, I) )
    kl = -0.5 * jnp.sum(1.0 + log_var - mu ** 2 - jnp.exp(log_var))

    return recon + beta * kl


def elbo_batch(
    model: LatentODE,
    padded: jax.Array,  # (B, T_max, d_embed)
    mask: jax.Array,    # (B, T_max) bool
    ts: jax.Array,      # (T_max,) float
    keys: jax.Array,    # (B, 2) PRNGKeys
    beta: float,
) -> jax.Array:
    def single(x, m, key):
        return elbo_single(model, x, m, ts, key, beta)

    per_sample = jax.vmap(single)(padded, mask, keys)
    return jnp.mean(per_sample)
