from PIL import Image
import numpy as np
import utils
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic
from einops import rearrange
from torchvision.transforms import functional as TF
from torch.nn import functional as F
from torch import nn
import pdb
import time

def sample_quadratic_bezier_curve(s, c, e, num_points=20):
    """
    Samples points from the quadratic bezier curves defined by the control points.
    Number of points to sample is num.

    Args:
        s (tensor): Start point of each curve, shape [N, 2].
        c (tensor): Control point of each curve, shape [N, 2].
        e (tensor): End point of each curve, shape [N, 2].
        num_points (int): Number of points to sample on every curve.

    Return:
       (tensor): Coordinates of the points on the Bezier curves, shape [N, num_points, 2]
    """
    N, _ = s.shape
    #t = torch.linspace(0., 1., num_points).to(s.device)
    t = torch.linspace(0., 1., num_points).type(torch.float32).to(s.device)
    
    t = torch.stack([t] * N, dim=0)
    s_x = s[..., 0:1]
    s_y = s[..., 1:2]
    e_x = e[..., 0:1]
    e_y = e[..., 1:2]
    c_x = c[..., 0:1]
    c_y = c[..., 1:2]
    x = c_x + (1. - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1. - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)
    return torch.stack([x, y], dim=-1)


def plot_tensor(img):
    final = img.detach().numpy()
    plt.imshow(final)

def find_mse(t1, t2):
    return np.mean((t1 - t2)**2)

def one2one_interp(array, size):
    res = torch.zeros((1, size[0], size[1], array.shape[-1])).to(array.device)
    for i in range(array.shape[-1]):
        res[...,i] = TF.resize(array[...,i],size=size,interpolation=TF.InterpolationMode.BILINEAR, antialias=False)        
    return res

# @torch.jit.script
def renderer(curve_points: torch.Tensor, locations: torch.Tensor, colors: torch.Tensor, widths: torch.Tensor,
                    H: int, W: int, K: int, canvas_color: float):
    dtype = torch.float32
    device = curve_points.device

    N, S, _ = curve_points.size()
    t_H = torch.linspace(0., float(H), int(H // 5)).to(device)
    t_W = torch.linspace(0., float(W), int(W // 5)).to(device)
    t_H = t_H.type(dtype=dtype)
    t_W = t_W.type(dtype=dtype)
    P_x, P_y = torch.meshgrid(t_H, t_W, indexing='ij')
    P = torch.stack([P_x, P_y], dim=-1)
    D_to_all_B_centers = torch.sum(torch.square(torch.unsqueeze(P, dim=-2) - locations), dim=-1)
    
    _, idcs = torch.topk(-D_to_all_B_centers, k=K, dim=-1)
    canvas_with_nearest_Bs = curve_points[idcs]
    canvas_with_nearest_Bs_colors = colors[idcs]
    canvas_with_nearest_Bs_bs = widths[idcs]
    
    H_, W_, r1, r2, r3 = canvas_with_nearest_Bs.shape
    canvas_with_nearest_Bs = rearrange(canvas_with_nearest_Bs.unsqueeze(0), 'B H W K S D -> B H W (K S D)')
    canvas_with_nearest_Bs = TF.resize(torch.permute(canvas_with_nearest_Bs, (0,3,1,2)), size=(H, W), interpolation=TF.InterpolationMode.NEAREST)
    canvas_with_nearest_Bs = torch.reshape(torch.permute(canvas_with_nearest_Bs, (0, 2, 3, 1)), (H, W, r1, r2, r3))
    
    H_, W_, r1, r2 = canvas_with_nearest_Bs_colors.shape
    canvas_with_nearest_Bs_colors = rearrange(canvas_with_nearest_Bs_colors.unsqueeze(0), 'B H W K D -> B H W (K D)')
    canvas_with_nearest_Bs_colors = TF.resize(torch.permute(canvas_with_nearest_Bs_colors, (0,3,1,2)), size=(H, W), interpolation=TF.InterpolationMode.NEAREST)
    canvas_with_nearest_Bs_colors = torch.reshape(torch.permute(canvas_with_nearest_Bs_colors, (0, 2, 3, 1)), (H, W, r1, r2))
    
    H_, W_, r1, r2 = canvas_with_nearest_Bs_bs.shape
    canvas_with_nearest_Bs_bs = rearrange(canvas_with_nearest_Bs_bs.unsqueeze(0), 'B H W K D -> B H W (K D)')
    canvas_with_nearest_Bs_bs = one2one_interp(canvas_with_nearest_Bs_bs, size=(H, W))
    canvas_with_nearest_Bs_bs = torch.reshape(canvas_with_nearest_Bs_bs, (H, W, r1, r2))
    
    
    #create full canvas
    t_H = torch.linspace(0., float(H), H).to(device)
    t_W = torch.linspace(0., float(W), W).to(device)
    t_H = t_H.type(dtype=dtype)
    t_W = t_W.type(dtype=dtype)
    P_x, P_y = torch.meshgrid(t_H, t_W, indexing='ij')
    P_full = torch.stack([P_x, P_y], dim=-1) # [H, W, 2]
    
    
    # Compute distance from every pixel on canvas to each (among nearest ones) line segment between points from curves
    canvas_with_nearest_Bs_a =  canvas_with_nearest_Bs[:,:,:,0:S-1,:]# start points of each line segment
    canvas_with_nearest_Bs_b =  canvas_with_nearest_Bs[:,:,:,1:S,:]# end points of each line segments
    canvas_with_nearest_Bs_b_a = canvas_with_nearest_Bs_b - canvas_with_nearest_Bs_a #[H, W, N, S - 1, 2]

    #pdb.set_trace()
    
    P_full_canvas_with_nearest_Bs_a = P_full.unsqueeze(-2).unsqueeze(-2) - canvas_with_nearest_Bs_a
    t = torch.sum(canvas_with_nearest_Bs_b_a * P_full_canvas_with_nearest_Bs_a, axis = -1) / torch.sum(torch.square(canvas_with_nearest_Bs_b_a), axis=-1)
    t = t.clamp(min=0, max=1)
    closest_points_on_each_line_segment = canvas_with_nearest_Bs_a + t.unsqueeze(-1) * canvas_with_nearest_Bs_b_a

    dist_to_closest_point_on_line_segment = \
        torch.sum(torch.square(P_full.unsqueeze(2).unsqueeze(2) - closest_points_on_each_line_segment), axis=-1)

    D = torch.amin(dist_to_closest_point_on_line_segment, dim=[-1, -2]) # [H, W]
    
    I_NNs_B_ranking = F.softmax(100000. * (1.0 / (1e-8 + torch.amin(dist_to_closest_point_on_line_segment, dim=[-1]))), dim=-1)
    
    
    I_colors = torch.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_colors, I_NNs_B_ranking)  # [H, W, 3]
    # pdb.set_trace()
    bs = torch.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_bs, I_NNs_B_ranking)  # [H, W, 1]
    
    bs_mask = torch.sigmoid(bs - D.unsqueeze(-1))
    
    canvas = torch.ones_like(I_colors, dtype=dtype) * canvas_color
    
    I = I_colors * bs_mask + (1 - bs_mask) * canvas
    
    return I

class BrushStrokeRenderer(nn.Module):
    def __init__(self, canvas_height, canvas_width, num_strokes=5000, samples_per_curve=10, strokes_per_pixel=20,
                 canvas_color='gray', length_scale=1.1, width_scale=.1, content_img=None):
        super().__init__()


        if canvas_color == 'gray':
            self.canvas_color = .5
        elif canvas_color == 'black':
            self.canvas_color = 0.
        elif canvas_color == 'noise':
            self.canvas_color = torch.rand(canvas_height, canvas_width, 3) * 0.1
        else:
            self.canvas_color = 1.

        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.num_strokes = num_strokes
        self.samples_per_curve = samples_per_curve
        self.strokes_per_pixel = strokes_per_pixel
        self.length_scale = length_scale
        self.width_scale = width_scale


        if content_img is not None:
            location, s, e, c, width, color = utils.initialize_brushstrokes(content_img, num_strokes,
                                                                            canvas_height, canvas_width,
                                                                            length_scale, width_scale)
        else:
            location, s, e, c, width, color = utils.initialize_brushstrokes(content_img, num_strokes,
                                                                            canvas_height, canvas_width,
                                                                            length_scale, width_scale, init='random')
        

        self.curve_s = torch.nn.Parameter(torch.from_numpy(np.array(s, 'float32')), requires_grad=True)
        self.curve_e = torch.nn.Parameter(torch.from_numpy(np.array(e, 'float32')), requires_grad=True)
        self.curve_c = torch.nn.Parameter(torch.from_numpy(np.array(c, 'float32')), requires_grad=True)
        self.color = torch.nn.Parameter(torch.from_numpy(color), requires_grad=True)
        self.location = torch.nn.Parameter(torch.from_numpy(np.array(location, 'float32')), requires_grad=True)
        self.width = torch.nn.Parameter(torch.from_numpy(np.array(width, 'float32')), requires_grad=True)

    def forward(self):
        #t1 = time.time()
        curve_points = sample_quadratic_bezier_curve(s=self.curve_s + self.location,
                                                     e=self.curve_e + self.location,
                                                     c=self.curve_c + self.location,
                                                     num_points=self.samples_per_curve)
        
        
        # renderer(curve_points: torch.Tensor, locations: torch.Tensor, colors: torch.Tensor, widths: torch.Tensor,
        #             H: int, W: int, K: int, canvas_color: float)
        canvas = renderer(curve_points, self.location, self.color, self.width,
                                 self.canvas_height, self.canvas_width, self.strokes_per_pixel, self.canvas_color)
        
        #t2 = time.time()
        #print(t2-t1)
        #pdb.set_trace()

        return canvas