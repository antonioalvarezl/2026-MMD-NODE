import copy
import time
import torch
import torch.nn as nn


def probability_cross_entropy(x_pred, x_obs):
    """Cross-entropy ``-sum(q * log(p))`` for simplex-valued trajectories."""
    if x_pred.shape != x_obs.shape:
        raise ValueError("Predicted and observed trajectories must have the same shape")
    eps = torch.finfo(x_pred.dtype).eps
    return -(x_obs * torch.log(x_pred.clamp_min(eps))).sum(dim=-1).mean()


class Trainer:
    """
    Trainer class for optimizing Neural ODE models to interpolate time series.
    
    Fits predicted simplex trajectories to observed mixture weights.
    """
    
    def __init__(self, model, optimizer, device, loss_str="MSE",
                 print_freq=10, max_epochs=1000,
                 early_stopping_min_delta=1e-5,
                 early_stopping_patience=200):
        """
        Initialize trainer.
        
        Args:
            model: Neural ODE model
            optimizer: PyTorch optimizer
            device: Training device
            loss_str: ``MSE``, ``L1`` or probability cross-entropy ``CE``
            print_freq: Print frequency for training progress
            max_epochs: Maximum training epochs
            tol: Absolute tolerance for early stopping
            tolrel: Relative tolerance for early stopping
            patience: Epochs to wait before early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Configure loss function
        loss_name = "MSE" if loss_str is None else str(loss_str).upper()
        if loss_name == 'MSE':
            self.loss_func = nn.MSELoss()
        elif loss_name == 'L1':
            self.loss_func = nn.L1Loss()
        elif loss_name in {'CE', 'CROSS_ENTROPY'}:
            self.loss_func = probability_cross_entropy
        else:
            raise ValueError(f"Unknown loss function: {loss_str}")
        
        self.print_freq = print_freq
        self.max_epochs = max_epochs
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        
        self.best_loss = float('inf')
        self.best_state = None
        self.loss_history = []
        self.best_epoch = None
        self.epochs_run = 0
        self.stop_reason = "not_started"

    def _compute_total_loss(self, x_pred, x_obs):
        """Compute the selected trajectory loss."""
        return self.loss_func(x_pred, x_obs)

    def _should_stop_early(self, patience_counter):
        """Stop only after the full patience without a material improvement."""
        return patience_counter >= self.early_stopping_patience

    def train(self, t_train, x_obs, initial_state, train_indices=None):
        """
        Train the Neural ODE model.
        
        Args:
            t_train: Time points tensor [n_timesteps]
            x_obs: Observed trajectory [n_timesteps, batch_size, state_dim]
            initial_state: Initial state [batch_size, state_dim]
        
        Returns:
            Best training loss
        """
        self.model.train()
        
        # Move data to device
        t_train = t_train.to(self.device)
        x_obs = x_obs.to(self.device)
        initial_state = initial_state.to(self.device)
        if train_indices is not None:
            train_indices = torch.as_tensor(
                train_indices, dtype=torch.long, device=self.device
            )
            if train_indices.ndim != 1 or len(train_indices) == 0:
                raise ValueError("train_indices must be a non-empty vector")
        
        patience_counter = 0
        monitored_best = float('inf')
        start_time = time.time()
        
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            x_pred = self.model(initial_state, eval_times=t_train)
            if train_indices is None:
                total_loss = self._compute_total_loss(x_pred, x_obs)
            else:
                total_loss = self._compute_total_loss(
                    x_pred[train_indices], x_obs[train_indices]
                )

            # ``total_loss`` belongs to the current (pre-update) parameters.
            # Record the matching state before taking an optimizer step.
            current_loss = total_loss.item()
            self.loss_history.append(current_loss)
            
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1

            if current_loss < monitored_best - self.early_stopping_min_delta:
                monitored_best = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            
            # Print progress
            if epoch % self.print_freq == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:4d} - Loss: {current_loss:.10f} - LR: {lr:.6f}")
            
            # Check early stopping
            if self._should_stop_early(patience_counter):
                self.stop_reason = "early_stopping"
                print(f"Early stopping at epoch {epoch}: no improvement of at least "
                      f"{self.early_stopping_min_delta} for "
                      f"{self.early_stopping_patience} epochs.")
                break

            # Backward pass
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.epochs_run = len(self.loss_history)
        if self.stop_reason == "not_started":
            self.stop_reason = "maximum_epochs_reached"

        # The state produced by the final optimizer step has not yet been
        # evaluated. Evaluate it once so every candidate state is paired with
        # the loss that it actually attains.
        with torch.no_grad():
            final_pred = self.model(initial_state, eval_times=t_train)
            if train_indices is None:
                final_loss = self._compute_total_loss(final_pred, x_obs).item()
            else:
                final_loss = self._compute_total_loss(
                    final_pred[train_indices], x_obs[train_indices]
                ).item()
        if final_loss < self.best_loss:
            self.best_loss = final_loss
            self.best_state = copy.deepcopy(self.model.state_dict())
            # This is the post-update state following the final recorded
            # pre-update objective, so it has its own checkpoint index.
            self.best_epoch = self.epochs_run + 1
        
        # Load best model state
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        fit_time = time.time() - start_time
        print(f"Training completed. Best Loss: {self.best_loss:.10f}")
        print(f"Fitting time: {fit_time:.4f} sec")
        
        return self.best_loss

    def evaluate(self, t_eval, x_obs, initial_state):
        """
        Evaluate model without parameter updates.
        
        Args:
            t_eval: Time points for evaluation
            x_obs: Observed trajectory  
            initial_state: Initial state
        
        Returns:
            Loss value
        """
        self.model.eval()
        with torch.no_grad():
            t_eval = t_eval.to(self.device)
            x_obs = x_obs.to(self.device)
            initial_state = initial_state.to(self.device)
            
            x_pred = self.model(initial_state, eval_times=t_eval)
            total_loss = self._compute_total_loss(x_pred, x_obs)
            
        return total_loss.item()
