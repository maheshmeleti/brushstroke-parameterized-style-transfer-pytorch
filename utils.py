import os
import torch
import numpy as np
from skimage.segmentation import slic
from scipy.spatial import ConvexHull
import requests
from PIL import Image
import torchvision.transforms as transforms

def clusters_to_strokes(segments, img, H, W, sec_scale=0.001, width_scale=1):
    segments += np.abs(np.min(segments))
    num_clusters = np.max(segments)
    clusters_params = {'center': [],
                       's': [],
                       'e': [],
                       'bp1': [],
                       'bp2': [],
                       'num_pixels': [],
                       'stddev': [],
                       'width': [],
                       'color_rgb': []
                       }

    for cluster_idx in range(num_clusters + 1):
        cluster_mask = segments == cluster_idx
        if np.sum(cluster_mask) < 5: continue
        cluster_mask_nonzeros = np.nonzero(cluster_mask)

        cluster_points = np.stack((cluster_mask_nonzeros[0], cluster_mask_nonzeros[1]), axis=-1)
        try:
            convex_hull = ConvexHull(cluster_points)
        except:
            continue

        # find the two points (pixels) in the cluster that have the largest distance between them
        border_points = cluster_points[convex_hull.simplices.reshape(-1)]
        dist = np.sum((np.expand_dims(border_points, axis=1) - border_points) ** 2, axis=-1)
        max_idx_a, max_idx_b = np.nonzero(dist == np.max(dist))
        point_a = border_points[max_idx_a[0]]
        point_b = border_points[max_idx_b[0]]
        # compute the two intersection points of the line that goes orthogonal to point_a and point_b
        v_ba = point_b - point_a
        v_orth = np.array([v_ba[1], -v_ba[0]])
        m = (point_a + point_b) / 2.0
        n = m + 0.5 * v_orth
        p = cluster_points[convex_hull.simplices][:, 0]
        q = cluster_points[convex_hull.simplices][:, 1]
        u = - ((m[..., 0] - n[..., 0]) * (m[..., 1] - p[..., 1]) - (m[..., 1] - n[..., 1]) * (m[..., 0] - p[..., 0])) \
            / ((m[..., 0] - n[..., 0]) * (p[..., 1] - q[..., 1]) - (m[..., 1] - n[..., 1]) * (p[..., 0] - q[..., 0]))
        intersec_idcs = np.logical_and(u >= 0, u <= 1)
        intersec_points = p + u.reshape(-1, 1) * (q - p)
        intersec_points = intersec_points[intersec_idcs]

        width = np.sum((intersec_points[0] - intersec_points[1]) ** 2)

        if width == 0.0: continue

        clusters_params['s'].append(point_a / img.shape[:2])
        clusters_params['e'].append(point_b / img.shape[:2])
        clusters_params['bp1'].append(intersec_points[0] / img.shape[:2])
        clusters_params['bp2'].append(intersec_points[1] / img.shape[:2])
        clusters_params['width'].append(np.sum((intersec_points[0] - intersec_points[1]) ** 2))

        clusters_params['color_rgb'].append(np.mean(img[cluster_mask], axis=0))
        center_x = np.mean(cluster_mask_nonzeros[0]) / img.shape[0]
        center_y = np.mean(cluster_mask_nonzeros[1]) / img.shape[1]
        clusters_params['center'].append(np.array([center_x, center_y]))
        clusters_params['num_pixels'].append(np.sum(cluster_mask))
        clusters_params['stddev'].append(np.mean(np.std(img[cluster_mask], axis=0)))

    for key in clusters_params.keys():
        clusters_params[key] = np.array(clusters_params[key])

    N = clusters_params['center'].shape[0]

    stddev = clusters_params['stddev']
    rel_num_pixels = 5 * clusters_params['num_pixels'] / np.sqrt(H * W)

    location = clusters_params['center']
    num_pixels_per_cluster = clusters_params['num_pixels'].reshape(-1, 1)
    s = clusters_params['s']
    e = clusters_params['e']
    cluster_width = clusters_params['width']

    location[..., 0] *= H
    location[..., 1] *= W
    s[..., 0] *= H
    s[..., 1] *= W
    e[..., 0] *= H
    e[..., 1] *= W

    s -= location
    e -= location

    color = clusters_params['color_rgb']

    c = (s + e) / 2. + np.stack([np.random.uniform(low=-1, high=1, size=[N]),
                                 np.random.uniform(low=-1, high=1, size=[N])],
                                axis=-1)

    sec_center = (s + e + c) / 3.
    s -= sec_center
    e -= sec_center
    c -= sec_center

    rel_num_pix_quant = np.quantile(rel_num_pixels, q=[0.3, 0.99])
    width_quant = np.quantile(cluster_width, q=[0.3, 0.99])
    rel_num_pixels = np.clip(rel_num_pixels, rel_num_pix_quant[0], rel_num_pix_quant[1])
    cluster_width = np.clip(cluster_width, width_quant[0], width_quant[1])
    width = width_scale * rel_num_pixels.reshape(-1, 1) * cluster_width.reshape(-1, 1)
    s, e, c = [x * sec_scale for x in [s, e, c]]

    location, s, e, c, width, color = [x.astype(np.float32) for x in [location, s, e, c, width, color]]

    return location, s, e, c, width, color

def batch_set_value(params, values):
    """
    Sets values of a tensor to another.

    :param params:
        a :class:`torch.Tensor`.
    :param values:
        a :class:`numpy.ndarray` of the same shape as `params`.
    :return:
        ``None``.
    """

    for p, v in zip(params, values):
        p.data.copy_(torch.from_numpy(v).data)


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


def projection(z):
    x = z[..., 0]
    y = z[..., 1]
    return torch.stack([x ** 2, y ** 2, x * y], dim=-1)


def download_file(url, name):
    if not os.path.exists(name):
        #logger.info(f'Downloading {url} into {name}...')

        filedir, filename = os.path.split(name)
        os.makedirs(filedir, exist_ok=True)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        ckpt_file_temp = name + '.temp'

        received = 0
        i = 0
        with open(ckpt_file_temp, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                received += len(data)
                file.write(data)
                percentage = 100. * received / total_size_in_bytes
                if i * 25 <= int(percentage) <= i * 27:
                    #logger.info(f'Downloaded {int(percentage)}%')
                    i += 1

        if total_size_in_bytes != 0 and received != total_size_in_bytes:
            #logger.error('An error occurred while downloading, please try again.')
            if os.path.exists(ckpt_file_temp):
                os.remove(ckpt_file_temp)
        else:
            os.rename(ckpt_file_temp, name)


def image_loader(image_name, img_size, device):
    loader = lambda imsize: transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()]) 
    
    image = Image.open(image_name)
    image = loader(img_size)(image).unsqueeze(0)
    image = image.to(device, torch.float)
    return image


def initialize_brushstrokes(content_img, num_strokes, canvas_height, canvas_width, sec_scale, width_scale, init='sp'):
    if init == 'random':
        # Brushstroke colors
        color = np.random.rand(num_strokes, 3)

        # Brushstroke widths
        width = np.random.rand(num_strokes, 1) * width_scale

        # Brushstroke locations
        location = np.stack([np.random.rand(num_strokes) * canvas_height, np.random.rand(num_strokes) * canvas_width],
                            axis=-1)

        # Start point for the Bezier curves
        s = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # End point for the Bezier curves
        e = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # Control point for the Bezier curves
        c = np.stack([np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_height,
                      np.random.uniform(low=-1, high=1, size=num_strokes) * canvas_width], axis=-1)

        # Normalize control points
        sec_center = (s + e + c) / 3.0
        s, e, c = [x - sec_center for x in [s, e, c]]
        s, e, c = [x * sec_scale for x in [s, e, c]]
    else:
        segments = slic(content_img,
                        n_segments=num_strokes,
                        min_size_factor=0.02,
                        max_size_factor=4.,
                        compactness=2,
                        sigma=1,
                        start_label=0)

        location, s, e, c, width, color = clusters_to_strokes(segments,
                                                              content_img,
                                                              canvas_height,
                                                              canvas_width,
                                                              sec_scale=sec_scale,
                                                              width_scale=width_scale)
        
        return location, s, e, c, width, color