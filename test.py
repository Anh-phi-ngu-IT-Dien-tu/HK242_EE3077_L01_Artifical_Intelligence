import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from sklearn.cluster import DBSCAN
import time
import threading
from multiprocessing import Process
from sklearn.linear_model import LinearRegression
import cmath
import open3d as o3d

flag_process = 0
n = 3
k = 35
dist_ref = 0.01
d = 700
bestFit = None
bestErr = 10000
threshold = 35
model = LinearRegression()
kernel = np.ones((5, 5), np.uint8)


class AppState:
    def __init__(self, *args, **kwargs):
        # self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

# Lop AppState() bao gom thong so huong camera, dich chuyen, khoang cach va trang thai cua chuot
# cung cap thuoc tinh truy cap ma tran xoay va diem tru
state = AppState()

# Cau hinh depth va color stream, cau hinh luong du lieu
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Yeu cau depth camera co cam bien mau!")
    exit(0)

# luong du lieu (16-bit depth), 30 fps va luong du lieu (8-bit moi kenh mau RGB), 30 fps
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

w, h = depth_intrinsics.width, depth_intrinsics.height

pc = rs.pointcloud()

# Tao bo loc giam kich thuoc du lieu
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()


# def findOutliers(X, inliers, outliers):
#     # Kiểm tra nếu inliers và X không rỗng
#     if (len(X) == 0) or (len(inliers) == 0):
#         return outliers
#
#     # Chuyển outliers thành danh sách Python nếu ban đầu là mảng numpy
#     if isinstance(outliers, np.ndarray):
#         outliers = outliers.tolist()
#
#     for i in X:
#         if i not in inliers:
#             # Thêm phần tử vào outliers dưới dạng mảng
#             outliers.append(i)  # Nếu i là một mảng hoặc list
#
#     # Chuyển lại thành numpy array sau khi append
#     outliers = np.array(outliers)
#     return outliers

def findOutliers(X, inliers):
    # Chuyển thành tập hợp để loại bỏ trùng lặp
    set_inliers = {tuple(row) for row in inliers}
    set_X = {tuple(row) for row in X}

    # Tìm outliers bằng phép trừ tập hợp
    set_outliers = set_X - set_inliers

    # Chuyển lại thành numpy array
    outliers = np.array(list(set_outliers))
    return outliers


# chuyen doi mang vector 3D thanh mang diem 2D
def project(v):  # "v" mang cac vector 3D
    v = np.array(v)
    h, w = out.shape[:2]
    view_aspect = float(h) / w

    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
               (w * view_aspect, h) + (w / 2.0, h / 2.0)

    # loai bo cac diem qua gan
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


# tai tao hieu ung, goc nhin va vi tri trong khong gian 3D
def view(v):
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


# ve doan thang 3D tu diem "pt1" den "pt2" len hinh anh "out"
def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]

    if np.isnan(p0).any() or np.isnan(p1).any():
        return

    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))

    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)

    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


# ve luoi tren mat phang xz
def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    # pos: vi tri luoi trong khong gian 3D
    # rotation: ma tran xoay ap dung cho luoi
    # size: kich thuoc luoi, n: so duong ke luoi
    # s: tinh toan khoang cach giua moi duong ke trong luoi, s2: nua kich thuoc luoi
    pos = np.array(pos)

    s = size / float(n)
    s2 = 0.5 * size
    # truc x
    for i in range(0, n + 1):
        x = -s2 + i * s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    # truc z
    for i in range(0, n + 1):
        z = -s2 + i * s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


# ve truc toa do 3D
def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


# ve khoi lap phuong (frustum) cua may anh len hinh anh out
def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        # lay toa do 3D goc trai, goc phai tren cung, goc phai, goc trai duoi cung frustum
        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


# ve diem pointclouds cua dam may diem 3D sang hinh 2D, sap xep tu xa den gan
def pointcloud(out, verts, texcoords, color, painter=True):
    if painter:
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5 ** state.decimate

    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T

    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]

    if painter:
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T

    np.clip(u, 0, ch - 1, out=u)
    np.clip(v, 0, cw - 1, out=v)

    out[i[m], j[m]] = color[u[m], v[m]]


def direction_control(angle):
    pi = math.pi
    angle_d = -1 * angle * 180 / pi  # Chuyển radian → độ

    hour_map = [
        (180, "9"), (135, "10"), (105, "11"), (75, "12"),
        (45, "1"), (15, "2"), (-1, "3")  # Góc nhỏ hơn 15° là 3h
    ]

    for threshold, hour in hour_map:
        if angle_d >= threshold:
            return hour
    return "none"  # Không xác định được hướng


def calculate_angle_from_center(target, shape):

    center = (shape[1] // 2, shape[0])  # Trung tâm cạnh dưới ảnh
    delta_x = target[0] - center[0]
    delta_y = target[1] - center[1]
    angle = np.arctan2(delta_y, delta_x)  # Tính góc (radian)
    return angle

def find_distances_min_clusters(cluster_centers_3d):
    # distances_min_clusters = cluster_centers_3d[0][2]
    point_nearest = [0,0,0]
    #print(cluster_centers_3d)
    # for cluster_center in cluster_centers_3d:
    #     if cluster_center[2] <= distances_min_clusters:
    #         distances_min_clusters = cluster_center[2]
    #         point_nearest = cluster_center
    distances_min_clusters = [np.linalg.norm(np.array(point_nearest)-np.array(cluster_center))
                             for cluster_center in cluster_centers_3d]
    point_nearest_idx = np.argmin(distances_min_clusters)
    point_nearest = cluster_centers_3d[point_nearest_idx]
    print(f"point_nearest:{point_nearest}")
    point_nearest_2d = project([point_nearest])
    print(f"point_nearest_2d:{point_nearest_2d}")
    distances_min_cluster = np.min(distances_min_clusters)
    print(f"distances_min_clusters:{distances_min_cluster}")

    # tra ve khoang cach, vi tri trai hay phai, huong mui gio cua vat gan nhat
    return distances_min_cluster, point_nearest_2d
# Tạo một tập hợp các màu sắc cố định
fixed_colors = [
    [255, 255, 0],  # Vàng
    [255, 0, 255],  # Tím
    [255, 0, 0],  # Đỏ
    [0, 255, 0],  # Lục
    [0, 255, 255],  # Cyan
    [0, 0, 255]  # Lam
]


# Áp dụng DBSCAN
def DBSCAN_segmentation(verts, w, h):
    segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
    segmented_image = cv2.resize(segmented_image, (648, 380))
    if len(verts) == 0:

        return segmented_image, {}
    # Lấy z
    points_2d = verts[:, :2]

    eps_custom = 0.07  # Điều chỉnh giá trị này để kiểm soát số lượng cụm
    dbscan = DBSCAN(eps=eps_custom, min_samples=200)
    clusters = dbscan.fit_predict(points_2d)

    # Lọc ra các cụm duy nhất (loại bỏ cụm nhiễu, được gán nhãn -1)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters != -1]
    cluster_centers_3d = []
    segmented_image = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)
    cluster_color_dict = {}
    num_clusters = len(unique_clusters)
    print(f"Số lượng cụm: {num_clusters}")
    # Gán màu sắc cho mỗi nhóm cluster
    for cluster_index in unique_clusters:
        color = fixed_colors[len(cluster_color_dict) % len(fixed_colors)]
        cluster_color_dict[cluster_index] = color
        mask = clusters == cluster_index
        points_in_cluster = verts[mask]
        points_2d = project(points_in_cluster[:, :3])
        in_img_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        points_2d = points_2d[in_img_mask]
        segmented_image[points_2d[:, 1].astype(int), points_2d[:, 0].astype(int)] = color
        cluster_center_3d = np.mean(points_in_cluster, axis=0) if points_in_cluster.size > 0 else np.array([np.nan, np.nan, np.nan])
        # print(f"cluster_center_3d:{cluster_center_3d}")
        cluster_centers_3d.append(cluster_center_3d)
        center_2d = project(cluster_centers_3d)
        print(f"center_2d:{center_2d}")

        # print(f"cluster_centers_3d:{cluster_centers_3d}")
        #print(cluster_centers_3d)
        if not cluster_centers_3d:
            print("Khong co vat the")
        else:
            # Tiếp tục xử lý nếu mảng không rỗng
            # print(verts_clusters)
            decimal_number_meter, point_nearest_2d = find_distances_min_clusters(cluster_centers_3d)
            print(f"point_nearest_2d:{point_nearest_2d}")
            if (point_nearest_2d is not None):
                center = [point_nearest_2d[:, 1].astype(int), point_nearest_2d[:, 0].astype(int)]
                new_org = (int(center[0] * (848 / 648)), int(center[1] * (380 / 480)))
                print(f"new_org:{new_org}")
                cv2.circle(segmented_image, new_org, 2, (255,255,255), thickness=2)
                # Chuyển đổi từ mét sang centimet
            decimal_number_cm = decimal_number_meter * 100

            # Làm tròn đến 0 chữ số thập phân
            temp_rounded_decimal_number_cm = round(decimal_number_cm, 0)
            print(f"Distance: {temp_rounded_decimal_number_cm}")

    segmented_image = cv2.resize(segmented_image, (648, 380))

    return segmented_image, cluster_color_dict


out = np.empty((h, w, 3), dtype=np.uint8)
flag_speak = True

while True:
    if not state.paused:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        depth_intrinsics = rs.video_stream_profile(
            depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        if state.color:
            mapped_frame, color_source = color_frame, color_image
        else:
            mapped_frame, color_source = depth_frame, depth_colormap

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()

        verts_int = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords_int = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Loc theo khoang cach
        min_distance = 0.2
        max_distance = 1
        # Tinh khoang cach cac diem
        distances = np.linalg.norm(verts_int, axis=1)
        # print(distances)
        # index của những điểm năm trong khoảng
        valid_indices = np.where((distances >= min_distance) & (distances <= max_distance))[0]

        step = 4  # cu 15 diem, giu lai 1 diem pointcloud
        verts = verts_int[valid_indices][::step]
        texcoords = texcoords_int[valid_indices][::step]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)

        # Áp dụng voxel down-sampling
        pcd = pcd.voxel_down_sample(voxel_size=0.008)

        # Define a bounding box for cropping (min_bound, max_bound)
        # Let's say we want to crop the region where x, y, z are between certain values
        min_bound = np.array([-0.2, -0.2, -1])  # Minimum corner of the box
        max_bound = np.array([0.2, 0.2, 1])  # Maximum corner of the box

        # Create a 3D bounding box using min_bound and max_bound
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # Crop the point cloud using the bounding box
        cropped_pcd = pcd.crop(bbox)

        # Visualize the cropped point cloud
        # o3d.visualization.draw_geometries([cropped_pcd])

        # Lấy lại các điểm đã được downsampled
        verts_downsampled = np.asarray(cropped_pcd.points)
        verts_downsampled = np.array(verts_downsampled)
        print("-" * 50)
        image_width = out.shape[1]
        image_height = out.shape[0]
        # Phân đoạn pointcloud bằng K-means clustering
        segmented_image, cluster_color_dict = DBSCAN_segmentation(verts_downsampled, image_width, image_height)
        color_image = cv2.resize(color_image, (648, 380))
        img_comb = cv2.hconcat([segmented_image, color_image])
        cv2.imshow("Segmented Image", img_comb)
        if cv2.waitKey(1) == ord('q'):
            break
pipeline.stop()
