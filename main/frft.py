import torch
import torch.nn as nn
import math

def cconvm(N, s, device=None):
    """Create a circulant convolution matrix."""
    if device is None:
        device = s.device
    M = torch.zeros((N, N), device=device)
    dum = s
    for i in range(N):
        M[:, i] = dum
        dum = torch.roll(dum, 1)
    return M

def dis_s(N, app_ord, device=None):
    """Calculate the discrete S matrix."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    app_ord = int(app_ord / 2)
    s = torch.cat((torch.tensor([0, 1], device=device), 
                   torch.zeros(N-1-2*app_ord, device=device), 
                   torch.tensor([1], device=device)))
    
    S = cconvm(N, s, device) + torch.diag((torch.fft.fft(s)).real)
    
    p = N
    r = math.floor(N/2)
    P = torch.zeros((p, p), device=device)
    P[0, 0] = 1
    even = 1 - (p % 2)
    
    for i in range(1, r-even+1):
        P[i, i] = 1/(2**(1/2))
        P[i, p-i] = 1/(2**(1/2))
    
    if even:
        P[r, r] = 1
    
    for i in range(r+1, p):
        P[i, i] = -1/(2**(1/2))
        P[i, p-i] = 1/(2**(1/2))
    
    CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)
    
    C2 = CS[0:math.floor(N/2+1), 0:math.floor(N/2+1)]
    S2 = CS[math.floor(N/2+1):N, math.floor(N/2+1):N]
    
    ec, vc = torch.linalg.eig(C2)
    es, vs = torch.linalg.eig(S2)
    
    ec = ec.real
    vc = vc.real
    es = es.real
    vs = vs.real
    
    qvc = torch.vstack((vc, torch.zeros([math.ceil(N/2-1), math.floor(N/2+1)], device=device)))
    SC2 = P @ qvc  # Even Eigenvector of S
    
    qvs = torch.vstack((torch.zeros([math.floor(N/2+1), math.ceil(N/2-1)], device=device), vs))
    SS2 = P @ qvs  # Odd Eigenvector of S
    
    idx = torch.argsort(-ec)
    SC2 = SC2[:, idx]
    
    idx = torch.argsort(-es)
    SS2 = SS2[:, idx]
    
    if N % 2 == 0:
        S2C2 = torch.zeros([N, N+1], device=device)
        SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0], 1), device=device)])
        S2C2[:, range(0, N+1, 2)] = SC2
        S2C2[:, range(1, N, 2)] = SS2
        S2C2 = S2C2[:, torch.arange(S2C2.size(1)) != N-1]
    else:
        S2C2 = torch.zeros([N, N], device=device)
        S2C2[:, range(0, N+1, 2)] = SC2
        S2C2[:, range(1, N, 2)] = SS2
    
    return S2C2

def dfrtmtrx(N, a, device=None):
    """Create the discrete FRFT matrix for order a."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Approximation order
    app_ord = 2
    
    Evec = dis_s(N, app_ord, device)
    Evec = Evec.to(dtype=torch.complex64)
    
    even = 1 - (N % 2)
    l = torch.tensor(list(range(0, N-1)) + [N-1+even], device=device)
    
    f = torch.diag(torch.exp(-1j * math.pi/2 * a * l))
    F = N**(1/2) * torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
    
    return F

def frft2d(image, order=0.8):
    """
    Calculate the 2D Fractional Fourier Transform of an image.
    
    Args:
        image: Input image tensor of shape [batch_size, channels, height, width]
        order: The fractional order of the transform (default: 0.5)
        
    Returns:
        The FRFT of the input image
    """
    device = image.device
    N, C, H, W = image.shape
    
    # Calculate FRFT matrices for height and width dimensions
    h_matrix = dfrtmtrx(H, order, device)
    w_matrix = dfrtmtrx(W, order, device)
    
    # Replicate matrices for batch and channel dimensions
    h_matrix = torch.repeat_interleave(h_matrix.unsqueeze(dim=0), repeats=C, dim=0)
    h_matrix = torch.repeat_interleave(h_matrix.unsqueeze(dim=0), repeats=N, dim=0)
    
    w_matrix = torch.repeat_interleave(w_matrix.unsqueeze(dim=0), repeats=C, dim=0)
    w_matrix = torch.repeat_interleave(w_matrix.unsqueeze(dim=0), repeats=N, dim=0)
    
    # Apply FFT shift, the FRFT matrices, and inverse FFT shift
    matrix = torch.fft.fftshift(image, dim=(2, 3)).to(dtype=torch.complex64)
    out = torch.matmul(h_matrix, matrix)
    out = torch.matmul(out, w_matrix)
    out = torch.fft.fftshift(out, dim=(2, 3))
    
    return out

def ifrft2d(image, order=0.8):
    """
    Calculate the 2D Inverse Fractional Fourier Transform of an image.
    
    Args:
        image: Input image tensor of shape [batch_size, channels, height, width]
        order: The fractional order of the transform (default: 0.5)
        
    Returns:
        The IFRFT of the input image
    """
    # For the inverse, we use the negative order
    return frft2d(image, -order)

def process_image(image_path, order=0.5):
    """
    Process an image with FRFT and visualize results.
    
    Args:
        image_path: Path to the input image
        order: The fractional order of the transform (default: 0.5)
        
    Returns:
        The FRFT result
    """
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = img_tensor.to(device)
    
    # Calculate FRFT
    frft_result = frft2d(img_tensor, order)
    
    # Calculate magnitude and phase
    magnitude = torch.abs(frft_result)
    phase = torch.angle(frft_result)
    
    # Convert to numpy for visualization
    magnitude_np = magnitude.cpu().squeeze().numpy()
    phase_np = phase.cpu().squeeze().numpy()
    
    # Log scale for better visualization
    magnitude_log = np.log1p(magnitude_np)
    
    # Normalize for visualization
    def normalize(img):
        return (img - img.min()) / (img.max() - img.min())
    
    magnitude_norm = normalize(magnitude_log)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_norm, cmap='viridis')
    plt.title(f'FRFT Magnitude (order={order})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(phase_np, cmap='hsv')
    plt.title(f'FRFT Phase (order={order})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return frft_result

# Example usage:
