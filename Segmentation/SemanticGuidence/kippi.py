import torch
from torch.optim import SGD
import numpy as np
from scipy.spatial import Delaunay
import os
import cv2

# 优化扰动参数，然后每次更新线段
# 采样后三角剖分
# Define the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PI = torch.tensor(3.141592653589793, dtype=torch.float32, device=device)
# Define hyperparameters
theta_max = 3 * (torch.pi / 180)  # Convert theta_max to radians as per the article
d_max = 1
lambda_reg = 0.8  # Regularization parameter
torch.random.manual_seed(1234)


# 在直线上采样点用于三角剖分
def sample_points_on_segments(line_segments, sample_interval=10):
    sampled_points = []
    segment_indices = []
    for i, segment in enumerate(line_segments):
        x1, y1, x2, y2 = segment
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)
        num_samples = int(np.ceil(length / sample_interval))

        # Create linear interpolations for x and y
        x_samples = np.linspace(x1, x2, num_samples)
        y_samples = np.linspace(y1, y2, num_samples)

        # Collect points and corresponding segment indices
        for x, y in zip(x_samples, y_samples):
            sampled_points.append([x, y])
            segment_indices.append(i)
    return np.array(sampled_points), np.array(segment_indices)


# 三角剖分的实现
def triangulate_points(points):
    """
    Perform Delaunay triangulation on a set of points.

    Args:
    points (np.ndarray): An array of point coordinates of shape (n_points, n_dimensions).
                         For 2D triangulation, this would be (n_points, 2).

    Returns:
    Delaunay: An object representing the Delaunay triangulation of the input points.
    """
    # Create a Delaunay object which computes the triangulation
    delaunay_tri = Delaunay(points)
    return delaunay_tri


def rotate_line_segments(line_segments, angle_deltas):
    new_segments = np.zeros_like(line_segments)
    for i, angle in enumerate(angle_deltas):
        # Extract the segment
        seg = line_segments[i]
        x1, y1, x2, y2 = seg
        # Calculate the midpoint of the segment
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0
        # Create the rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        # Subtract midpoint, apply rotation, and add midpoint back for both points
        point1 = np.array([x1 - midpoint_x, y1 - midpoint_y])
        point2 = np.array([x2 - midpoint_x, y2 - midpoint_y])
        rotated_point1 = rotation_matrix @ point1
        rotated_point2 = rotation_matrix @ point2
        new_segments[i, :2] = rotated_point1 + np.array([midpoint_x, midpoint_y])
        new_segments[i, 2:] = rotated_point2 + np.array([midpoint_x, midpoint_y])
    return new_segments


def translate_line_segments(line_segments, offset_deltas):
    new_segments = np.zeros_like(line_segments)
    for i in range(line_segments.shape[0]):
        # Extract the endpoints of the line segment
        x1, y1, x2, y2 = line_segments[i]
        # Calculate the direction vector of the line segment
        direction_vector = np.array([x2 - x1, y2 - y1])
        # Calculate the perpendicular vector (normal to the line segment)
        perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
        # Normalize the perpendicular vector
        unit_perpendicular = perpendicular_vector / np.linalg.norm(perpendicular_vector)
        # Translate both endpoints along the perpendicular vector by the offset distance
        new_segments[i, :2] = [x1, y1] + unit_perpendicular * offset_deltas[i]
        new_segments[i, 2:] = [x2, y2] + unit_perpendicular * offset_deltas[i]
    return new_segments


def calculate_alpha_ij_vectorized(line_segments):
    # Vectorized calculation of direction vectors for all line segments
    direction_vectors = line_segments[:, 2:4] - line_segments[:, 0:2]
    # Normalize the direction vectors
    norms = torch.norm(direction_vectors, dim=1, keepdim=True)
    unit_vectors = direction_vectors / norms
    # Calculate dot products between all pairs of unit vectors
    dot_products = torch.mm(unit_vectors, unit_vectors.t())
    # Clamp the dot products to avoid numerical errors outside the [-1, 1] range
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angles using the arccos of the dot products
    angles = torch.acos(dot_products)
    # Convert angles to degrees if needed (optional)
    # angles = angles * (180 / PI)
    return angles


def calculate_mu_ij_vectorized(seg_lines, points, point_indices, tri):
    n_segments = seg_lines.shape[0]
    mu_list = list(set({}) for i in range(n_segments))
    # 遍历三角形
    for _ in tri.simplices:
        result = [point_indices[i] for i in _]
        mu_list[result[0]].add(result[1])
        mu_list[result[0]].add(result[2])
        mu_list[result[1]].add(result[0])
        mu_list[result[1]].add(result[2])
        mu_list[result[2]].add(result[0])
        mu_list[result[2]].add(result[1])
    return mu_list


def vectorized_distance_between_almost_parallel_segments(segments, mu, tolerance=1e-5):
    n = segments.shape[0]
    distances = torch.zeros((n, n), device=segments.device)
    # 分别计算所有线段的方向向量
    direction_vectors = segments[:, 2:4] - segments[:, 0:2]
    # 规范化方向向量
    norms = torch.norm(direction_vectors, dim=1, keepdim=True)
    unit_vectors = direction_vectors / norms
    # 计算所有单位向量对之间的点积，得到余弦值
    cosines = torch.mm(unit_vectors, unit_vectors.t())
    # 检查近似平行性：余弦值接近1表示平行
    parallelism_mask = torch.abs(cosines) > 1 - tolerance
    # 遍历所有可能的线段对
    for i in range(len(mu)):
        for j in mu[i]:
            if parallelism_mask[i, j]:
                # 计算线段i和线段j之间的距离
                p1 = segments[i, :2]
                p3 = segments[j, :2]
                # 连接向量
                connecting_vector = p3 - p1
                # 计算垂直距离
                normal_vector = torch.tensor([-unit_vectors[i, 1], unit_vectors[i, 0]], device=segments.device)
                distance = torch.abs(torch.dot(connecting_vector, normal_vector))
                distances[i, j] = distance
                distances[j, i] = distance  # 距离矩阵是对称的
    return distances


def dx_theta(theta_perturbations, theta_max):
    return torch.sum((theta_perturbations / theta_max) ** 2) / len(theta_perturbations)


def dx_dist(x_perturbations, d_max):
    return torch.sum((x_perturbations / d_max) ** 2) / len(x_perturbations)


def vx_theta(line_segments, theta_max, x_perturbation, alpha, mu):
    n = line_segments.shape[0]
    mu_sum = 0
    V = 0
    for i in range(n):
        for j in mu[i]:
            mu_ij = 0
            # Calculate the relative angle α_ij between line-segments i and j
            alpha_ij = alpha[i, j]
            # Calculate θ_ij based on α_ij as per the article
            if alpha_ij in [PI / 4., PI * 3. / 4.] or alpha_ij in [-PI / 4., PI / 4.]:
                theta_ij = alpha_ij % PI
            else:
                theta_ij = (alpha_ij - PI / 2) % PI
            if torch.abs(theta_ij) < 2 * theta_max:
                mu_ij = 1
            v_ij = mu_ij * torch.abs(theta_ij - x_perturbation[i] + x_perturbation[j])
            mu_sum += mu_ij
            V += v_ij
    return V / (4. * theta_max * mu_sum)


def vx_dist(line_segments, d_max, x_perturbation, distances, mu, alpha):
    n = line_segments.size(0)
    V = 0
    for i in range(n):
        for j in mu[i]:
            v_ij = 0
            distance = distances[i, j]
            alpha_ij = alpha[i, j]
            if alpha_ij in [PI / 4., PI * 3. / 4.] or alpha_ij in [-PI / 4., PI / 4.]:
                theta_ij = alpha_ij % PI
            else:
                theta_ij = (alpha_ij - PI / 2) % PI
            if distance > 0 and torch.abs(theta_ij) < 2 * theta_max and distances < 2 * d_max:
                v_ij = torch.abs(distance - x_perturbation[i] + x_perturbation[j])
            V += v_ij
    return V / (4. * d_max)


# Define the total energy function U(x) to be minimized
def total_loss(line_segments, d_max, theta_max, theta_perturbations, x_perturbations, alpha, distances,
               mu, lable=None):
    if lable == 'angle':
        angle_loss = (1 - lambda_reg) * dx_theta(theta_perturbations, theta_max) + \
                     lambda_reg * vx_theta(line_segments, theta_max, theta_perturbations, alpha=alpha, mu=mu)
        return angle_loss
    elif lable == 'location':
        dist_loss = (1 - lambda_reg) * dx_dist(x_perturbations, d_max) + \
                    lambda_reg * vx_dist(line_segments, d_max, x_perturbations, distances, mu=mu, alpha=alpha)
        return dist_loss
    else:
        angle_loss = (1 - lambda_reg) * dx_theta(theta_perturbations, theta_max) + \
                     lambda_reg * vx_theta(line_segments, theta_max, theta_perturbations, alpha=alpha, mu=mu)
        dist_loss = (1 - lambda_reg) * dx_dist(x_perturbations, d_max) + \
                    lambda_reg * vx_dist(line_segments, d_max, x_perturbations, distances, mu=mu, alpha=alpha)
        return angle_loss + dist_loss


def vis_folder(line_segments, theta_root_path, x_root_path, image, image_save_path):
    theta_perturbations_files = [theta_root_path + _ for _ in os.listdir(theta_root_path)]
    x_perturbations_files = [x_root_path + _ for _ in os.listdir(x_root_path)]
    temp_lines = line_segments
    model = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    assert len(theta_perturbations_files) == len(x_perturbations_files)
    for i in range(len(theta_perturbations_files)):
        theta = np.loadtxt(theta_perturbations_files[i])
        x = np.loadtxt(x_perturbations_files[i])
        lines_rotate = rotate_line_segments(temp_lines, theta)
        lines_transfer = translate_line_segments(lines_rotate, x)
        temp_lines = lines_transfer
        draws = model.drawSegments(image, temp_lines)
        cv2.imwrite(image_save_path + 'iter{:04d}lines.jpg'.format(i), draws)


def vis_img(line_segments, theta_file, x_file, image, image_save_path, index):
    temp_lines = line_segments
    model = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
    theta = np.loadtxt(theta_file)
    x = np.loadtxt(x_file)
    lines_rotate = rotate_line_segments(temp_lines, theta)
    lines_transfer = translate_line_segments(lines_rotate, x)
    temp_lines = lines_transfer
    draws = model.drawSegments(image, temp_lines)
    cv2.imwrite(image_save_path + 'iter{:04d}lines.jpg'.format(index), draws)


def main(load_best=True, vis=True):
    root_path = 'F:/SegAndAlign/MVS/line_detect/'
    best_save_path = root_path + 'results_03/'
    theta_save_path = best_save_path + 'theta/'
    x_save_path = best_save_path + 'x/'
    vis_path = best_save_path + 'vis/'
    if not os.path.exists(theta_save_path):
        os.makedirs(theta_save_path)
    if not os.path.exists(x_save_path):
        os.makedirs(x_save_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    # 原始图像
    image_path = 'F:/SegAndAlign/MVS/source_images/DJI_20230516152238_0012_V.JPG'
    # 加载LSD检测到的线段
    lines_txt = 'F:/SegAndAlign/MVS/line_detect/DJI_20230516152238_0012_V.txt'
    line_seg_results = np.loadtxt(lines_txt, dtype=np.float32)
    # 在线段上进行采样
    # return [n_lines, n_samples+2, 2]
    sample_points, indices = sample_points_on_segments(line_segments=line_seg_results, sample_interval=10)
    # Delaunay 三角剖分
    points = sample_points.reshape(-1, 2)
    tri = triangulate_points(points)

    line_seg_results_tensor = torch.from_numpy(line_seg_results).to(device)
    # points = torch.from_numpy(points).to(device)
    # 随机产生一个初始扰动值
    if load_best:
        theta_perturbation = np.loadtxt(
            best_save_path + 'Best_theta_for_DJI_20230516152238_0012_V.txt',
            dtype=np.float32)
        theta_perturbation = torch.from_numpy(theta_perturbation).to(device).requires_grad_(True)
        x_perturbation = np.loadtxt(best_save_path + 'Best_x_for_DJI_20230516152238_0012_V.txt',
                                    dtype=np.float32)
        x_perturbation = torch.from_numpy(x_perturbation).to(device).requires_grad_(True)
    else:
        theta_perturbation = (torch.rand(len(line_seg_results_tensor), dtype=torch.float32,
                                         device=device) * 0.02 - 0.01) * theta_max
        x_perturbation = (torch.rand(len(line_seg_results_tensor), dtype=torch.float32,
                                     device=device) * 0.02 - 0.01) * d_max
        theta_perturbation.requires_grad_(True)
        x_perturbation.requires_grad_(True)

    # 预计算
    mu = calculate_mu_ij_vectorized(line_seg_results_tensor, points, indices, tri)
    alpha = calculate_alpha_ij_vectorized(line_seg_results_tensor)
    distances = vectorized_distance_between_almost_parallel_segments(line_seg_results_tensor, mu)

    optimizer = SGD([theta_perturbation, x_perturbation], lr=0.05, momentum=0.9)

    loss_ = 1000000
    # Optimization loop
    for epoch in range(200):  # Number of epochs
        optimizer.zero_grad()  # Zero out the gradients
        loss = total_loss(line_seg_results_tensor, d_max, theta_max,
                          theta_perturbation, x_perturbation,
                          alpha=alpha, distances=distances, mu=mu)
        loss.backward()  # Compute the gradient of the loss
        optimizer.step()  # Update the line segments
        with torch.no_grad():
            theta_perturbation.data.clamp(-theta_max, theta_max)
            x_perturbation.data.clamp(-d_max, d_max)

        print(f'Epoch {epoch}, Loss: {loss.item()}')
        if loss.item() < loss_:
            np.savetxt(
                best_save_path + 'Best_theta_for_DJI_20230516152238_0012_V.txt',
                theta_perturbation.detach().cpu().numpy())
            np.savetxt(
                best_save_path + 'Best_x_for_DJI_20230516152238_0012_V.txt',
                x_perturbation.detach().cpu().numpy())
            loss_ = loss.item()
            if vis:
                image = cv2.imread(image_path)
                vis_img(line_seg_results,
                        best_save_path + 'Best_theta_for_DJI_20230516152238_0012_V.txt',
                        best_save_path + 'Best_x_for_DJI_20230516152238_0012_V.txt', image,
                        best_save_path, 0)

        np.savetxt(
            theta_save_path + '{:04d}_Theta_for_DJI_20230516152238_0012_V.txt'.format(
                epoch), theta_perturbation.detach().cpu().numpy())
        np.savetxt(
            x_save_path + '{:04d}_X_for_DJI_20230516152238_0012_V.txt'.format(epoch),
            x_perturbation.detach().cpu().numpy())
        if vis:
            image = cv2.imread(image_path)
            black_image = np.zeros_like(image)
            vis_img(line_seg_results,
                    theta_save_path + '{:04d}_Theta_for_DJI_20230516152238_0012_V.txt'.format(
                        epoch),
                    x_save_path + '{:04d}_X_for_DJI_20230516152238_0012_V.txt'.format(
                        epoch), black_image, vis_path, epoch)


if __name__ == '__main__':
    main(load_best=False, vis=True)
    print('This is kippi.py...')
