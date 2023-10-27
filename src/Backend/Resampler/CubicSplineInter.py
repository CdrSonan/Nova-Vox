import torch

def h_poly(t:torch.Tensor) -> torch.Tensor:
    """returns Hermite basis functions for the cubic hermite splines used in cubic interpolation"""

    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

def interp(x:torch.Tensor, y:torch.Tensor, xs:torch.Tensor) -> torch.Tensor:
    """perform cubic interpolation using Hermite splines
    
    Arguments:
        x, y: Tensors of x and y values representing sample points on a curve. Should be the same length in dimension 0.
        
        xs: The position the interpolation should be performed at. Must be between the minimum and maximum of x.
        
    Returns:
        y value corresponding to xs, so the point [xs, y] lies on the curve defined by x and y
        
    cubic Hermite interpolation is not as accurate for audio as sinc interpolation and produces more artifacting in fourier space.
    Nonetheless, it is an excellent compromise between accuracy and speed."""

    
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    
def extrap(x:torch.Tensor, y:torch.Tensor, xs:torch.Tensor) -> torch.Tensor:
    derr = y[-1] - y[-2]
    derr2 = y[-1] - 2 * y[-2] + y[-3]
    largeY = y[-1] + (derr + 0. * derr2) * (torch.max(xs) - x[-1]) / (x[-1] - x[-2])
    derr = y[1] - y[0]
    derr2 = y[2] - 2 * y[1] + y[0]
    smallY = y[0] - (derr + 0. * derr2) * (x[0] - torch.min(xs)) / (x[1] - x[0])
    if torch.min(xs) < torch.min(x):
        x = torch.cat((torch.unsqueeze(torch.min(xs), 0), x), 0)
        y = torch.cat((torch.unsqueeze(smallY, 0), y), 0)
    if torch.max(xs) > torch.max(x):
        x = torch.cat((x, torch.unsqueeze(torch.max(xs), 0)), 0)
        y = torch.cat((y, torch.unsqueeze(largeY, 0)), 0)
    return interp(x, y, xs)
