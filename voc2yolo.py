import xml.etree.ElementTree as ET
import os
import glob

def convert_voc_to_yolo(xml_path, output_dir, class_mapping):
    """
    将 Pascal VOC 格式的 XML 文件转换为 YOLO 格式的 txt 文件。

    Args:
        xml_path: XML 文件的路径。
        output_dir: 输出 YOLO 格式 txt 文件的目录。
        class_mapping: 类别名称到 YOLO 类别 ID 的映射字典。
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    filename = os.path.splitext(os.path.basename(xml_path))[0] + '.txt'
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_mapping:
                print(f"Warning: Class '{class_name}' not found in class mapping. Skipping object.")
                continue
            class_id = class_mapping[class_name]

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # 转换为 YOLO 格式的中心点坐标和宽高
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def batch_convert(data_dirs, class_mapping):
    """
    批量转换 train, test, val 文件夹中的 XML 文件。

    Args:
        data_dirs: 包含 train, test, val 文件夹路径的列表。
        class_mapping: 类别名称到 YOLO 类别 ID 的映射字典。
        output_base_dir: 输出 YOLO 格式 txt 文件的根目录。
    """

    for data_dir in data_dirs:
        set_name = os.path.basename(data_dir)  # 获取 train, test, val 的名称
        xml_dir = os.path.join(data_dir, 'Annotations', 'xmls')
        output_dir = os.path.join(data_dir, 'labels')
        os.makedirs(output_dir, exist_ok=True)

        xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
        for xml_file in xml_files:
            convert_voc_to_yolo(xml_file, output_dir, class_mapping)
            print(f"Converted: {xml_file} -> {output_dir}")

# --- 使用示例 ---
# 假设你的数据集目录结构如下：
# dataset_root/
# ├── train/
# │   ├── Annotations/
# │   │   └── xmls/
# │   │       ├── 000001.xml
# │   │       ├── 000002.xml
# │   │       └── ...
# │   └── ...
# ├── test/
# │   ├── Annotations/
# │   │   └── xmls/
# │   │       └── ...
# │   └── ...
# └── val/
#     ├── Annotations/
#     │   └── xmls/
#     │       └── ...
#     └── ...

# 数据集根目录
dataset_root = './orange/'  # 请替换为你的数据集根目录

# train, test, val 文件夹路径
data_dirs = [
    os.path.join(dataset_root, 'train'),
    os.path.join(dataset_root, 'test'),
    os.path.join(dataset_root, 'val')
]

# 类别映射字典，请根据你的数据集进行修改
class_mapping = {
    'orange': 0,  # 例如，将 'orange' 映射到类别 ID 0
    # 添加其他类别...
}

# 输出 YOLO 格式 txt 文件的根目录
# output_base_dir = 'path/to/your/output_dir' # 请替换为你希望的输出目录

# 执行批量转换
batch_convert(data_dirs, class_mapping)

print("Conversion completed!")
