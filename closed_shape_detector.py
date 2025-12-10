"""
ComfyUI封闭图形检测节点
作者：运维工程师
功能：检测图像中的封闭图形，标记为白色，其他区域为黑色
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Any

class ClosedShapesDetector:
    """
    封闭图形检测节点
    检测图像中的封闭图形并生成对应的Mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        detection_methods = [
            "边缘检测+轮廓填充",
            "阈值分割+轮廓分析",
            "形态学操作+闭区域检测",
            "边缘闭合+区域填充",
            "多方法融合"
        ]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (detection_methods, {"default": "边缘检测+轮廓填充"}),
                "min_area": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "max_area": ("INT", {"default": 50000, "min": 100, "max": 1000000}),
                "output_format": (["mask", "outline", "both"], {"default": "mask"}),
            },
            "optional": {
                "edge_threshold_low": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 255.0}),
                "edge_threshold_high": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 255.0}),
                "binary_threshold": ("FLOAT", {"default": 128.0, "min": 1.0, "max": 255.0}),
                "morph_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21}),
                "close_gaps": ("BOOLEAN", {"default": True}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "filter_by_shape": ("BOOLEAN", {"default": False}),
                "shape_criteria": ("STRING", {
                    "multiline": True,
                    "default": "{\"min_aspect_ratio\": 0.2, \"max_aspect_ratio\": 5.0}"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("visualization", "mask", "info")
    FUNCTION = "detect_closed_shapes"
    CATEGORY = "image/processing/detection"
    DESCRIPTION = "检测图像中的封闭图形并生成Mask"
    
    def detect_closed_shapes(self, image: torch.Tensor, method: str, min_area: int, 
                            max_area: int, output_format: str, **kwargs):
        
        # 转换图像为numpy
        img_np = self.tensor_to_numpy(image[0])  # 取批次中的第一张
        
        # 灰度化
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 根据选择的方法检测封闭图形
        if method == "边缘检测+轮廓填充":
            mask = self.method_edge_contour(gray, **kwargs)
        elif method == "阈值分割+轮廓分析":
            mask = self.method_threshold_contour(gray, **kwargs)
        elif method == "形态学操作+闭区域检测":
            mask = self.method_morphology(gray, **kwargs)
        elif method == "边缘闭合+区域填充":
            mask = self.method_edge_closing(gray, **kwargs)
        elif method == "多方法融合":
            mask = self.method_fusion(gray, **kwargs)
        else:
            mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # 过滤面积
        mask = self.filter_by_area(mask, min_area, max_area)
        
        # 后处理
        if kwargs.get('close_gaps', True):
            mask = self.close_gaps(mask, kwargs.get('morph_kernel_size', 3))
        
        if kwargs.get('fill_holes', True):
            mask = self.fill_holes(mask)
        
        # 形状过滤
        if kwargs.get('filter_by_shape', False):
            shape_criteria = kwargs.get('shape_criteria', '{}')
            mask = self.filter_by_shape_criteria(mask, shape_criteria)
        
        # 创建可视化输出
        visualization = self.create_visualization(img_np, mask, output_format)
        
        # 生成信息
        info = self.generate_info(mask, method, min_area, max_area)
        
        # 转换为ComfyUI格式
        mask_tensor = self.numpy_to_mask(mask)
        vis_tensor = self.numpy_to_tensor(visualization)
        
        return (vis_tensor, mask_tensor, info)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将torch tensor转换为numpy数组"""
        img_np = tensor.cpu().numpy()
        
        # 转换为0-255范围
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        return img_np
    
    def numpy_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为torch tensor"""
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_np).float()
        
        # 确保正确的维度 (1, H, W, C)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def numpy_to_mask(self, mask_np: np.ndarray) -> torch.Tensor:
        """将numpy mask转换为torch mask"""
        if mask_np.dtype == np.uint8:
            mask_np = mask_np.astype(np.float32) / 255.0
        
        mask_tensor = torch.from_numpy(mask_np).float()
        
        # 确保正确的维度 (1, H, W)
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return mask_tensor
    
    def method_edge_contour(self, gray: np.ndarray, **kwargs) -> np.ndarray:
        """方法1: 边缘检测+轮廓填充"""
        # 边缘检测参数
        low_thresh = kwargs.get('edge_threshold_low', 50.0)
        high_thresh = kwargs.get('edge_threshold_high', 150.0)
        
        # 边缘检测
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            edges, 
            cv2.RETR_CCOMP,  # 检索所有轮廓并建立层次关系
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建空白mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if contours:
            # 填充所有轮廓（包括子轮廓）
            for i, cnt in enumerate(contours):
                # 检查是否是外部轮廓或孔洞
                if hierarchy[0][i][3] == -1:  # 外部轮廓
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
                else:  # 孔洞，填充为黑色（0）
                    cv2.drawContours(mask, [cnt], -1, 0, -1)
        
        return mask
    
    def method_threshold_contour(self, gray: np.ndarray, **kwargs) -> np.ndarray:
        """方法2: 阈值分割+轮廓分析"""
        # 阈值参数
        threshold = kwargs.get('binary_threshold', 128.0)
        
        # 自适应阈值
        if threshold > 0:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        # 形态学操作改善分割
        kernel_size = kwargs.get('morph_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建mask并填充
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if contours:
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] == -1:  # 外部轮廓
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return mask
    
    def method_morphology(self, gray: np.ndarray, **kwargs) -> np.ndarray:
        """方法3: 形态学操作+闭区域检测"""
        kernel_size = kwargs.get('morph_kernel_size', 3)
        
        # 使用梯度检测边缘
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 二值化
        _, binary = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
        
        # 闭操作连接边缘
        close_kernel = np.ones((kernel_size*2, kernel_size*2), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        
        # 查找并填充轮廓
        contours, _ = cv2.findContours(
            closed, 
            cv2.RETR_EXTERNAL,  # 只检测外部轮廓
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask
    
    def method_edge_closing(self, gray: np.ndarray, **kwargs) -> np.ndarray:
        """方法4: 边缘闭合+区域填充"""
        # 边缘检测
        low_thresh = kwargs.get('edge_threshold_low', 50.0)
        high_thresh = kwargs.get('edge_threshold_high', 150.0)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        
        # 形态学闭操作闭合边缘间隙
        kernel_size = kwargs.get('morph_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            closed_edges, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建mask并填充
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        return mask
    
    def method_fusion(self, gray: np.ndarray, **kwargs) -> np.ndarray:
        """方法5: 多方法融合"""
        # 应用多种方法
        mask1 = self.method_edge_contour(gray, **kwargs)
        mask2 = self.method_threshold_contour(gray, **kwargs)
        
        # 融合结果（逻辑或）
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学后处理
        kernel_size = kwargs.get('morph_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def filter_by_area(self, mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
        """根据面积过滤区域"""
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 创建新的mask
        filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
        
        if contours:
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                
                # 只处理外部轮廓
                if hierarchy[0][i][3] == -1 and min_area <= area <= max_area:
                    cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        return filtered_mask
    
    def close_gaps(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """闭合mask中的小间隙"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return closed
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """填充mask中的孔洞"""
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 如果找到轮廓，填充所有孔洞
        if contours:
            # 先复制一份mask
            filled = mask.copy()
            
            # 找到所有孔洞并填充
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] != -1:  # 孔洞
                    cv2.drawContours(filled, [cnt], -1, 255, -1)
            
            return filled
        
        return mask
    
    def filter_by_shape_criteria(self, mask: np.ndarray, shape_criteria: str) -> np.ndarray:
        """根据形状条件过滤区域"""
        import json
        
        try:
            criteria = json.loads(shape_criteria)
        except:
            criteria = {}
        
        # 默认条件
        min_aspect_ratio = criteria.get('min_aspect_ratio', 0.2)
        max_aspect_ratio = criteria.get('max_aspect_ratio', 5.0)
        min_circularity = criteria.get('min_circularity', 0.1)
        max_circularity = criteria.get('max_circularity', 1.0)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
        
        if contours:
            for i, cnt in enumerate(contours):
                # 只处理外部轮廓
                if hierarchy[0][i][3] == -1:
                    # 计算包围矩形
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # 计算长宽比
                    if w == 0:
                        continue
                    aspect_ratio = h / w
                    
                    # 计算圆形度
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # 应用过滤条件
                    if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                        min_circularity <= circularity <= max_circularity):
                        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
        
        return filtered_mask
    
    def create_visualization(self, original: np.ndarray, mask: np.ndarray, 
                           output_format: str) -> np.ndarray:
        """创建可视化输出"""
        if output_format == "mask":
            # 只显示mask
            visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        elif output_format == "outline":
            # 在原图上绘制轮廓
            visualization = original.copy()
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 绘制轮廓
            cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)
        
        else:  # "both"
            # 创建三通道的mask并叠加到原图
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            
            # 创建半透明的红色遮罩
            red_mask = np.zeros_like(original)
            red_mask[:, :, 2] = mask  # 红色通道
            
            # 叠加到原图
            visualization = cv2.addWeighted(original, 0.7, red_mask, 0.3, 0)
            
            # 添加轮廓
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(visualization, contours, -1, (0, 255, 0), 1)
        
        return visualization
    
    def generate_info(self, mask: np.ndarray, method: str, 
                     min_area: int, max_area: int) -> str:
        """生成检测信息"""
        # 统计检测到的区域
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        num_regions = len(contours)
        
        # 计算总面积
        total_area = 0
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_area += area
            areas.append(area)
        
        # 计算统计信息
        if areas:
            avg_area = np.mean(areas)
            max_detected_area = np.max(areas)
            min_detected_area = np.min(areas)
        else:
            avg_area = max_detected_area = min_detected_area = 0
        
        info = f"""
封闭图形检测报告:
=================
检测方法: {method}
检测到的区域数量: {num_regions}
总面积: {total_area:.0f} 像素
平均区域面积: {avg_area:.1f} 像素
最大区域面积: {max_detected_area:.0f} 像素
最小区域面积: {min_detected_area:.0f} 像素
面积过滤范围: {min_area} - {max_area} 像素
"""
        
        return info


class ConnectedComponentsDetector:
    """
    连通区域检测节点
    使用连通组件分析检测封闭区域
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["threshold", "edges", "adaptive"], {"default": "threshold"}),
                "connectivity": (["4-connect", "8-connect"], {"default": "8-connect"}),
                "min_size": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "label_colors": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "threshold": ("FLOAT", {"default": 128.0, "min": 0.0, "max": 255.0}),
                "canny_low": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 255.0}),
                "canny_high": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 255.0}),
                "block_size": ("INT", {"default": 11, "min": 3, "max": 101}),
                "c": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("visualization", "mask", "components_info")
    FUNCTION = "detect_components"
    CATEGORY = "image/processing/detection"
    DESCRIPTION = "使用连通组件分析检测封闭区域"
    
    def detect_components(self, image: torch.Tensor, method: str, connectivity: str,
                         min_size: int, label_colors: bool, **kwargs):
        
        # 转换图像
        img_np = self.tensor_to_numpy(image[0])
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 根据方法创建二值图像
        if method == "threshold":
            threshold = kwargs.get('threshold', 128.0)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        elif method == "edges":
            low_thresh = kwargs.get('canny_low', 50.0)
            high_thresh = kwargs.get('canny_high', 150.0)
            binary = cv2.Canny(gray, low_thresh, high_thresh)
        
        elif method == "adaptive":
            block_size = kwargs.get('block_size', 11)
            c = kwargs.get('c', 2.0)
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, c
            )
        
        # 反转二值图像（确保前景为白色）
        binary = cv2.bitwise_not(binary)
        
        # 设置连通性
        conn = 8 if connectivity == "8-connect" else 4
        
        # 进行连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=conn
        )
        
        # 创建彩色标签图像
        if label_colors:
            # 为每个标签分配随机颜色
            colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  # 背景为黑色
            
            colored_labels = colors[labels]
        else:
            # 使用灰度表示
            colored_labels = (labels * (255 // num_labels)).astype(np.uint8)
            colored_labels = cv2.cvtColor(colored_labels, cv2.COLOR_GRAY2RGB)
        
        # 创建mask（仅包含大于最小尺寸的区域）
        mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, num_labels):  # 跳过背景
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                mask[labels == i] = 255
        
        # 在原图上绘制边界框和中心点
        visualization = img_np.copy()
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # 获取边界框
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # 绘制边界框
                cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 绘制中心点
                center_x, center_y = map(int, centroids[i])
                cv2.circle(visualization, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # 添加标签文本
                cv2.putText(visualization, f"{i}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 生成信息
        info = self.generate_components_info(num_labels, stats, min_size)
        
        # 转换为ComfyUI格式
        mask_tensor = self.numpy_to_mask(mask)
        vis_tensor = self.numpy_to_tensor(visualization)
        
        return (vis_tensor, mask_tensor, info)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将torch tensor转换为numpy数组"""
        img_np = tensor.cpu().numpy()
        
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        return img_np
    
    def numpy_to_tensor(self, img_np: np.ndarray) -> torch.Tensor:
        """将numpy数组转换为torch tensor"""
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_np).float()
        
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def numpy_to_mask(self, mask_np: np.ndarray) -> torch.Tensor:
        """将numpy mask转换为torch mask"""
        if mask_np.dtype == np.uint8:
            mask_np = mask_np.astype(np.float32) / 255.0
        
        mask_tensor = torch.from_numpy(mask_np).float()
        
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        
        return mask_tensor
    
    def generate_components_info(self, num_labels: int, stats: np.ndarray, 
                               min_size: int) -> str:
        """生成连通组件信息"""
        # 统计有效区域
        valid_regions = 0
        total_area = 0
        areas = []
        
        for i in range(1, num_labels):  # 跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                valid_regions += 1
                total_area += area
                areas.append(area)
        
        # 计算统计信息
        if areas:
            avg_area = np.mean(areas)
            max_area = np.max(areas)
            min_area = np.min(areas)
        else:
            avg_area = max_area = min_area = 0
        
        info = f"""
连通组件分析报告:
=================
总标签数量: {num_labels}
有效区域数量: {valid_regions}
最小面积阈值: {min_size}
总面积: {total_area:.0f} 像素
平均区域面积: {avg_area:.1f} 像素
最大区域面积: {max_area:.0f} 像素
最小区域面积: {min_area:.0f} 像素
"""
        
        return info



