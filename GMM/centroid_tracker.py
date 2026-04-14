import numpy as np
from scipy.spatial import distance
# CentroidTracker để theo dõi các đối tượng dựa trên tọa độ tâm (centroid) của chúng trong các khung hình video.

class CentroidTracker:
    def __init__(self, max_distance=80, max_disappeared=30): # max_disappeared là 30 nghĩa là sau 30 frame ko thấy hay nhận diện đc người đó  nữa thì xóa
        self.next_object_id = 0                             #max_distance = 80 nghĩa là nếu centroid mới cách centroid cũ hơn 80 px thì ko coi là cùng 1 người nữa
        self.objects = {}
        self.disappeared = {}

        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
    # Hàm register : Gán ID mới -Lưu centroid - Reset số frame mất  
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    # Hàm deregister : Xóa đối tượng khỏi danh sách theo dõi ( k quan trọng đâu )
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    # Hàm update : Cập nhật vị trí các đối tượng dựa trên các hộp giới hạn mới được phát hiện ( tracking chính )
    def update(self, boxes):
        if len(boxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        # Tính toán centroid từ các hộp giới hạn mới
        input_centroids = np.zeros((len(boxes), 2), dtype="int")
        
        for (i, (x, y, w, h)) in enumerate(boxes):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0) 
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(tuple(c))
            return self.objects
        # chú ý phần này
        # So sánh centroid mới với centroid đã theo dõi
        object_ids = list(self.objects.keys()) # Lấy danh sách ID của các đối tượng hiện tại
        object_centroids = np.array(list(self.objects.values()))

        D = distance.cdist(object_centroids, input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            # nếu khoảng cách vượt quá max_distance, bỏ qua không ghép cặp (sẽ được tạo ID mới ở dưới)
            if D[row, col] > self.max_distance: 
                continue

            # nếu hợp lệ (khoảng cách < max_distance), cập nhật vị trí mới cho ID cũ đang theo dõi
            obj_id = object_ids[row]
            self.objects[obj_id] = tuple(input_centroids[col]) 
            self.disappeared[obj_id] = 0 # cập nhật lại số frame mất về 0 

            used_rows.add(row)
            used_cols.add(col)
        #kh quan trọng nhưng cũng đọc qua , khúc này là nếu object cũ mà ko thấy trong frame mới thì tăng biến disappeared lên 1, vượt quá max_disappeared thì xóa
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.register(tuple(input_centroids[col]))

        return self.objects
