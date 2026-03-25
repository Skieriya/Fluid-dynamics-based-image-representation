import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VISCOSITY = 0.02
ABSORPTION_STRENGTH = 20.0
INLET_VELOCITY = 1.0
EPOCHS = 1500
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
LAYERS = 4

def get_mnist_obstacle(digit_idx=0):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    image, label = dataset[digit_idx]
    absorption_map = image.squeeze().numpy()
    absorption_map = np.flipud(absorption_map).copy()
    return absorption_map, label

class FlowPINN(nn.Module):
    def __init__(self):
        super(FlowPINN, self).__init__()
        layers = []
        layers.append(nn.Linear(2, HIDDEN_DIM))
        layers.append(nn.Tanh())
        for _ in range(LAYERS):
            layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(HIDDEN_DIM, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

def physics_loss(model, x, y, absorption_tensor):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u, v, p = model(x, y)

    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    cont = u_x + v_y

    size = absorption_tensor.shape[0]
    xi = ((x + 1) / 2 * (size - 1)).long().clamp(0, size - 1)
    yi = ((y + 1) / 2 * (size - 1)).long().clamp(0, size - 1)
    sigma = absorption_tensor[yi, xi]

    momx = -p_x + VISCOSITY * (u_xx + u_yy) - ABSORPTION_STRENGTH * sigma * u
    momy = -p_y + VISCOSITY * (v_xx + v_yy) - ABSORPTION_STRENGTH * sigma * v

    return torch.mean(cont**2) + torch.mean(momx**2) + torch.mean(momy**2)

def boundary_loss(model, x_bc, y_bc, u_bc, v_bc):
    u_pred, v_pred, _ = model(x_bc, y_bc)
    return torch.mean((u_pred - u_bc)**2) + torch.mean((v_pred - v_bc)**2)

def run_mnist_simulation(digit_index=0):
    absorption_np, label = get_mnist_obstacle(digit_index)
    absorption_tensor = torch.tensor(absorption_np).to(device)

    n_int = 6000
    x_int = torch.FloatTensor(n_int, 1).uniform_(-1, 1).to(device)
    y_int = torch.FloatTensor(n_int, 1).uniform_(-1, 1).to(device)

    n_bc = 600
    x_in = torch.full((n_bc, 1), -1.0).to(device)
    y_in = torch.FloatTensor(n_bc, 1).uniform_(-1, 1).to(device)
    x_wall = torch.FloatTensor(n_bc, 1).uniform_(-1, 1).to(device)
    y_top = torch.full((n_bc, 1), 1.0).to(device)
    y_bot = torch.full((n_bc, 1), -1.0).to(device)

    x_bc = torch.cat([x_in, x_wall, x_wall])
    y_bc = torch.cat([y_in, y_top, y_bot])
    u_bc = torch.cat([torch.full((n_bc, 1), INLET_VELOCITY), torch.zeros((2*n_bc, 1))]).to(device)
    v_bc = torch.zeros((3*n_bc, 1)).to(device)

    model = FlowPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for e in range(EPOCHS):
        opt.zero_grad()
        lp = physics_loss(model, x_int, y_int, absorption_tensor)
        lb = boundary_loss(model, x_bc, y_bc, u_bc, v_bc)
        loss = lp + 100 * lb
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        g = 100
        xv = torch.linspace(-1, 1, g).to(device)
        yv = torch.linspace(-1, 1, g).to(device)
        X, Y = torch.meshgrid(xv, yv, indexing='ij')
        u_p, v_p, _ = model(X.reshape(-1, 1), Y.reshape(-1, 1))
        u = u_p.reshape(g, g).cpu().numpy()
        v = v_p.reshape(g, g).cpu().numpy()

    dx = 2.0 / g
    dv_dx = np.gradient(v, dx, axis=0)
    du_dy = np.gradient(u, dx, axis=1)
    curl = dv_dx - du_dy

    return absorption_np, u, v, curl, label

r1 = run_mnist_simulation(0)
r2 = run_mnist_simulation(1)

fig = plt.figure(figsize=(14, 6))

def add_plot(img, u, v, curl, label, row):
    plt.subplot(2, 4, 1 + row*4)
    plt.imshow(img, cmap="gray", origin="lower")
    plt.title(f"Digit {label}")

    sp = np.sqrt(u*u + v*v)
    plt.subplot(2, 4, 2 + row*4)
    plt.imshow(sp, cmap="jet", origin="lower")
    plt.title("Speed")

    Y, X = np.mgrid[-1:1:100j, -1:1:100j]
    plt.subplot(2, 4, 3 + row*4)
    plt.streamplot(X, Y, u.T, v.T, color='k', density=1)
    plt.imshow(img, cmap='gray', alpha=0.2, extent=[-1,1,-1,1], origin='lower')
    plt.title("Flow")

    plt.subplot(2, 4, 4 + row*4)
    plt.imshow(curl, cmap="seismic", origin="lower", vmin=-30, vmax=30)
    plt.title("Vorticity")

add_plot(*r1, 0)
add_plot(*r2, 1)

plt.tight_layout()
plt.show()
