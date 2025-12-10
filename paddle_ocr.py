import kornia
import torch
from paddleocr import PaddleOCR
import comfy.model_management


class OcrBoxMask:
    def __init__(self):
        print("OcrFunction init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'orc_box_mask'

    def orc_box_mask(self, images, text, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        masks = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            words = text.split(";")
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        # print(line[1][0])
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                if points[0][1] > 1:
                                    points[0][1] -= 1
                                if points[2][1] < shape[0] - 1:
                                    points[2][1] += 1
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    if x_min > 1:
                                        x_min -= 1
                                    if x_max < shape[1] - 1:
                                        x_max += 1

                                    mask[int(points[0][1]):int(points[2][1]), int(x_min):int(x_max)] = 1
                                    start = end
            masks.append(mask.unsqueeze(0))
        return (torch.cat(masks, dim=0),)


class OcrImageText:
    def __init__(self):
        print("OcrImageText init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {
            "required": {
                "images": ("IMAGE",),
                "lang": (lang_list, {"default": "ch"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'orc_image_text'

    def orc_image_text(self, images, lang):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

        text = ""
        last_text = ""
        for image in images:
            i = 255. * image.cpu().numpy()
            now_text = ""
            orc_ret = self.ocr.ocr(i, cls=False)
            for idx in range(len(orc_ret)):
                res = orc_ret[idx]
                if res is not None:
                    for line in res:
                        if line[1][0] != "":
                            now_text += line[1][0] + "\n"
            if now_text != "" and now_text != last_text:
                text += now_text + "\n"
                last_text = now_text
        return (text,)


class OcrBlur:
    def __init__(self):
        print("OcrBlur init")
        self.lang = "ch"
        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {"required":
            {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": ""}),
                "blur": ("INT", {"default": 255, "min": 3, "max": 8191, "step": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'orc_blur'

    def orc_blur(self, images, text, lang, blur):
        if lang != self.lang:
            self.lang = lang
            del self.ocr
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        new_images = []
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            shape = i.shape
            mask = torch.zeros((shape[0], shape[1]), dtype=torch.uint8)
            words = text.split(";")
            result = self.ocr.ocr(i, cls=False)
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        # print(line[1][0])
                        for word in words:
                            if word == "":
                                continue
                            if text == "" or line[1][0].find(word) >= 0:
                                text_line = line[1][0]
                                points = line[0]
                                if points[0][1] > 1:
                                    points[0][1] -= 1
                                if points[2][1] < shape[0] - 1:
                                    points[2][1] += 1
                                total_length = len(text_line)
                                start = 0
                                while text_line.find(word, start) >= 0:
                                    start = text_line.find(word, start)
                                    end = start + len(word)
                                    x_min = points[0][0] + start * (points[1][0] - points[0][0]) / total_length
                                    x_max = points[0][0] + end * (points[1][0] - points[0][0]) / total_length
                                    if x_min > 1:
                                        x_min -= 1
                                    if x_max < shape[1] - 1:
                                        x_max += 1

                                    mask[int(points[0][1]):int(points[2][1]), int(x_min):int(x_max)] = 1
                                    start = end

            # blur the image by mask
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2)
            blurred = image.clone()
            alpha = mask_floor(mask_unsqueeze(mask))
            alpha = alpha.expand(-1, 3, -1, -1)
            blurred = gaussian_blur(blurred, blur, 0)
            blurred = image + (blurred - image) * alpha
            new_images.append(blurred.permute(0, 2, 3, 1))
        return (torch.cat(new_images, dim=0),)


def gaussian_blur(image, radius: int, sigma: float = 0):
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    image = image.to(comfy.model_management.get_torch_device())
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma)).cpu()


def mask_floor(mask, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


def mask_unsqueeze(mask):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask



import numpy as np


class OcrAllTextMask:
    """
    识别并遮罩图像中的所有文本区域
    无需输入文本，自动检测所有OCR识别到的文本
    """
    def __init__(self):
        print("OcrAllTextMask init")
        self.lang = "ch"
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        except Exception as e:
            print(f"[PaddleOCR] 初始化失败: {e}")
            print(f"[PaddleOCR] 建议手动下载模型文件到: C:\\Users\\Aiden\\.paddleocr\\")
            self.ocr = None

    @classmethod
    def INPUT_TYPES(self):
        lang_list = ["ch", "latin", "arabic", "cyrillic", "devanagari", "en"]
        return {
            "required": {
                "lang": (lang_list, {"default": "ch"}),
                "images": ("IMAGE",),
                "padding": ("INT", {"default": 2, "min": 0, "max": 50, "step": 1}),
                "mask_all_text": ("BOOLEAN", {"default": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "exclude_words": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "IMAGE", "LIST")
    RETURN_NAMES = ("mask", "detected_text", "debug_image", "bounding_boxes")
    FUNCTION = 'ocr_all_text_mask'
    
    CATEGORY = "image/ocr"

    def ocr_all_text_mask(self, images, lang, padding, mask_all_text, confidence_threshold, exclude_words=""):
        # 更新语言设置
        if lang != self.lang:
            self.lang = lang
            if hasattr(self, 'ocr') and self.ocr is not None:
                del self.ocr
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            except Exception as e:
                print(f"[PaddleOCR] 重新初始化失败: {e}")
                empty_mask = torch.zeros((len(images), 256, 256), dtype=torch.uint8)
                return (empty_mask, "", images, [])
        
        # 检查OCR是否已初始化
        if self.ocr is None:
            print("[PaddleOCR] OCR未初始化，无法处理图像")
            empty_mask = torch.zeros((len(images), 256, 256), dtype=torch.uint8)
            return (empty_mask, "", images, [])
        
        # 解析要排除的单词
        exclude_words_list = []
        if exclude_words:
            exclude_words_list = [w.strip().lower() for w in exclude_words.split("\n") if w.strip()]
        
        masks = []
        all_text = []
        debug_images = []
        all_bounding_boxes = []
        
        for batch_idx, image in enumerate(images):
            # 转换为numpy图像
            i = 255. * image.cpu().numpy()
            shape = i.shape
            h, w = shape[:2]
            
            # 创建mask
            mask = torch.zeros((h, w), dtype=torch.uint8)
            
            # 运行OCR
            result = self.ocr.ocr(i, cls=False)
            
            batch_text = []
            batch_bboxes = []
            
            # 创建调试图像（可视化文本区域）
            debug_img = image.clone()
            
            for idx in range(len(result)):
                res = result[idx]
                if res is not None:
                    for line in res:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # 检查置信度阈值
                        if confidence < confidence_threshold:
                            continue
                        
                        # 检查是否在排除列表中
                        if exclude_words_list and text.lower() in exclude_words_list:
                            continue
                        
                        # 保存识别的文本
                        batch_text.append(text)
                        
                        # 获取文本的边界框（4个点）
                        points = line[0]
                        
                        # 转换为边界框
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        
                        # 计算边界框坐标
                        x_min = max(0, int(min(xs)) - padding)
                        x_max = min(w, int(max(xs)) + padding)
                        y_min = max(0, int(min(ys)) - padding)
                        y_max = min(h, int(max(ys)) + padding)
                        
                        # 确保边界框有效
                        if x_max > x_min and y_max > y_min:
                            batch_bboxes.append({
                                "text": text,
                                "confidence": confidence,
                                "bbox": [x_min, y_min, x_max, y_max],
                                "points": points
                            })
                            
                            # 如果mask_all_text为True，则遮罩整个文本区域
                            if mask_all_text:
                                mask[y_min:y_max, x_min:x_max] = 1
                            else:
                                # 只遮罩文本本身（更精确的边界）
                                # 创建一个更精确的多边形mask
                                poly_mask = self._create_polygon_mask(points, h, w, padding)
                                mask = torch.logical_or(mask, poly_mask).to(torch.uint8)
            
            # 将文本组合为字符串
            text_str = "\n".join(batch_text)
            all_text.append(text_str)
            
            # 保存bounding boxes
            all_bounding_boxes.append(batch_bboxes)
            
            # 添加到mask列表
            masks.append(mask.unsqueeze(0))
            
            # 创建调试图像（用红色框标记文本区域）
            debug_img = self._draw_bounding_boxes(debug_img, batch_bboxes)
            debug_images.append(debug_img)
        
        # 组合所有批次的mask
        final_mask = torch.cat(masks, dim=0) if masks else torch.zeros((len(images), 256, 256), dtype=torch.uint8)
        
        # 组合所有批次的调试图像
        final_debug_img = torch.cat(debug_images, dim=0) if debug_images else images
        
        # 组合所有批次的文本
        all_text_str = "\n--- Batch Separator ---\n".join(all_text)
        
        return (final_mask, all_text_str, final_debug_img, all_bounding_boxes)
    
    def _create_polygon_mask(self, points, h, w, padding):
        """
        从多边形点创建mask
        """
        import cv2
        
        # 创建空白mask
        mask_np = np.zeros((h, w), dtype=np.uint8)
        
        # 将点转换为整数
        pts = np.array(points, dtype=np.int32)
        
        # 绘制多边形
        cv2.fillPoly(mask_np, [pts], 1)
        
        # 如果需要，添加padding
        if padding > 0:
            kernel = np.ones((padding*2+1, padding*2+1), np.uint8)
            mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        
        # 转换回torch tensor
        return torch.from_numpy(mask_np).to(torch.uint8)
    
    def _draw_bounding_boxes(self, image_tensor, bboxes):
        """
        在图像上绘制边界框（用于调试）
        """
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 转换为numpy进行绘制
        img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        
        try:
            import cv2
            
            for bbox_info in bboxes:
                x_min, y_min, x_max, y_max = bbox_info["bbox"]
                text = bbox_info["text"]
                confidence = bbox_info["confidence"]
                
                # 绘制矩形框
                cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                
                # 绘制文本
                text_display = f"{text[:20]}({confidence:.2f})"
                font_scale = max(0.5, min(1.0, 20.0 / len(text)))
                cv2.putText(img_np, text_display, (x_min, y_min - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)
        
        except ImportError:
            print("警告: 未安装cv2，无法绘制边界框")
        
        # 转换回torch tensor
        img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
        return img_tensor
