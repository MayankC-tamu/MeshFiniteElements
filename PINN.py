import torch
import torch.nn as nn
import numpy as np
import tqdm

# 1. Define the Neural Network Architecture
class PoissonPINN(nn.Module):
    def __init__(self):
        super(PoissonPINN, self).__init__()
        # 3 hidden layers with 32 neurons each. 
        # Tanh is crucial because we need continuous second derivatives.
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# 2. Compute the PDE Residual using Automatic Differentiation
def compute_pde_loss(model, x, y):
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    u = model(x, y)
    
    # First derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    # Target source term: f(x, y) = 2*(x^2 + y^2 - 2)
    f = 2 * (x**2 + y**2 - 2)
    
    # Residual of the Poisson equation: u_xx + u_yy - f = 0
    residual = u_xx + u_yy - f
    return torch.mean(residual**2)

# 3. Generate Training Data (Domain: [-1, 1] x [-1, 1])
# Interior collocation points
N_interior = 1000
x_int = torch.rand(N_interior, 1) * 2 - 1  # scale to [-1, 1]
y_int = torch.rand(N_interior, 1) * 2 - 1

# Boundary points where u = 0
N_boundary = 200
# Edges at x = -1 and x = 1
x_b1 = torch.cat([torch.full((N_boundary, 1), -1.0), torch.full((N_boundary, 1), 1.0)], dim=0)
y_b1 = torch.rand(2 * N_boundary, 1) * 2 - 1

# Edges at y = -1 and y = 1
x_b2 = torch.rand(2 * N_boundary, 1) * 2 - 1
y_b2 = torch.cat([torch.full((N_boundary, 1), -1.0), torch.full((N_boundary, 1), 1.0)], dim=0)

# Combine boundary coordinates
x_b = torch.cat([x_b1, x_b2], dim=0)
y_b = torch.cat([y_b1, y_b2], dim=0)
u_b_true = torch.zeros_like(x_b)  # u = 0 on the boundary

# 4. Training Loop
model = PoissonPINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Starting training...")
for epoch in range(3001):
    optimizer.zero_grad()
    
    # Compute losses
    loss_pde = compute_pde_loss(model, x_int, y_int)
    
    u_b_pred = model(x_b, y_b)
    loss_bc = torch.mean((u_b_pred - u_b_true)**2)
    
    # Total physics-informed loss (heavily weighting the boundary helps convergence)
    total_loss = loss_pde + 10.0 * loss_bc
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | PDE Loss: {loss_pde.item():.6f} | BC Loss: {loss_bc.item():.6f}")


E_local = []
def u(x, y):
    return (x**2 - 1) * (y**2 - 1)

print("\n--- Verification ---")
with tqdm.tqdm(total=160801, desc=f"Computing L2 Error") as pbar:
        for x in np.arange(-1, 1.005, 0.005):
            for y in np.arange(-1, 1.005, 0.005):
                u_value = model(torch.tensor([[x]], dtype=torch.float32), torch.tensor([[y]], dtype=torch.float32)).item()
                U_exact = u(x, y)
                E_local.append((u_value - U_exact)**2)
                #print(f"Point: ({x}, {y}), FEM Value: {u_value}, Exact Value: {U_exact}, Squared Error: {(u_value - U_exact)**2}")
                pbar.update(1)

print("L2 Error:", np.sqrt(np.mean(E_local)))
