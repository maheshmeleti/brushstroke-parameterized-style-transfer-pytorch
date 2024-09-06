import os
import sys
import torch
import torch
import torch.optim as optim
from torchvision.transforms import functional as F
import utils
import losses
from render_tools import BrushStrokeRenderer
from tqdm import tqdm
import pdb
from PIL import Image
import numpy as np
import cv2
import argparse

vgg_weight_file = '/scratch/umeleti/code/style/pytorch_brushstroke/brushstroke-parameterized-style-transfer/vgg_weights/vgg19_weights_normalized.h5'

# desired depth layers to compute style/content losses :
bs_content_layers = ['conv4_1', 'conv5_1']
bs_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
px_content_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
px_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def run_stroke_style_transfer(num_steps=100, style_weight=3., content_weight=1., tv_weight=0.008, curv_weight=4):
    vgg_loss = losses.StyleTransferLosses(vgg_weight_file, content_img, style_img,
                                          bs_content_layers, bs_style_layers, scale_by_y=True)
    vgg_loss.to(device).eval()

    # brush stroke init
    bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                      canvas_color, length_scale, width_scale,
                                      content_img=content_img[0].permute(1, 2, 0).cpu().numpy())
    bs_renderer.to(device)

    optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                            bs_renderer.curve_c, bs_renderer.width], lr=1e-1)
    optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

    
    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()
        optimizer_color.zero_grad()
        input_img = bs_renderer()
        input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
        content_score, style_score = vgg_loss(input_img)

        style_score *= style_weight
        content_score *= content_weight
        tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                           bs_renderer.curve_e, K=10)
        curv_score = curv_weight * losses.curvature_loss(bs_renderer.curve_s, bs_renderer.curve_e, bs_renderer.curve_c)
        loss = style_score + content_score + tv_score + curv_score
        loss.backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                              bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
        optimizer.step()
        style_score.backward(inputs=[bs_renderer.color])
        optimizer_color.step()

        # plot some stuffs
        # mon.plot('stroke style loss', style_score.item())
        # mon.plot('stroke content loss', content_score.item())
        # mon.plot('stroke tv loss', tv_score.item())
        # mon.plot('stroke curvature loss', curv_score.item())
        # if mon.iter % mon.print_freq == 0:
        #     mon.imwrite('stroke stylized', input_img)

    with torch.no_grad():
        return bs_renderer()

# if __name__ == '__main__':
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str, required=True, help='Content image name')
parser.add_argument('--style', type=str, required=True, help='Style image name')
parser.add_argument('--nstrokes', type=int, required=True, help='Number of brush strokes')
args = parser.parse_args()

imgs_path = '/images/'
content_img_file = os.path.join(imgs_path, args.content)
style_img_file = os.path.join(imgs_path, args.style)
#output_name = f'{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}'
output_name = 'result.png'

imsize = 512
content_img = utils.image_loader(content_img_file, imsize, device)
style_img = utils.image_loader(style_img_file, 224, device)

canvas_color =  'gray'
num_strokes = args.nstrokes
samples_per_curve = 20
brushes_per_pixel = 20
_, _, H, W = content_img.shape
canvas_height = H
canvas_width = W
length_scale = 1.1
width_scale = 0.1

canvas = run_stroke_style_transfer()

np_array = canvas.detach().cpu().numpy()
# np_array_ = (np_array - np.min(np_array))/(np.max(np_array)-np.min(np_array))*255
np_array_ = np.clip(np_array, 0, 1)
image = Image.fromarray(np.uint8(np_array_*255.0))
image.save(output_name)
# cv2.imwrite(output_name, np_array_[:,:,::-1])
