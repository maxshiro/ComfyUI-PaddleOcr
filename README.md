# ComfyUI-PaddleOcr
Nodes related to PaddleOCR OCR
- Inspire by [PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/) 

基于 PaddleOCR 的 ComfyUI 文字识别与处理节点插件。

## 功能节点

### 1. OcrBoxMask
**功能**：根据指定文本在图像中生成对应文字区域的遮罩。

**输入参数**：
- `images` (IMAGE)：输入图像（支持批量）
- `lang` (SELECT)：识别语言
  - 可选值：`ch`（中文）、`latin`、`arabic`、`cyrillic`、`devanagari`、`en`
  - 默认值：`ch`
- `text` (STRING)：要查找的文本，多个用分号`;`分隔

**输出**：
- `mask` (MASK)：文字区域的二值遮罩

---

### 2. OcrBlur
**功能**：对图像中指定的文字区域进行高斯模糊处理。

**输入参数**：
- `images` (IMAGE)：输入图像
- `lang` (SELECT)：识别语言（同 OcrBoxMask）
- `text` (STRING)：要模糊的文本
- `blur` (INT)：模糊半径（奇数，3-8191，默认 255）

**输出**：
- `image` (IMAGE)：文字区域模糊后的图像

---

### 3. OcrImageText
**功能**：从图像中提取所有文字内容。

**输入参数**：
- `images` (IMAGE)：输入图像
- `lang` (SELECT)：识别语言（同 OcrBoxMask）

**输出**：
- `text` (STRING)：识别出的文字内容

---

### 4. OcrAllTextMask
**功能**：自动检测图像中的所有文字区域，提供遮罩、调试图像和详细识别信息。

**输入参数**：
- `images` (IMAGE)：输入图像
- `lang` (SELECT)：识别语言
- `padding` (INT)：边界框扩展像素（0-50，默认 2）
- `mask_all_text` (BOOLEAN)：是否遮罩整个矩形区域（True）或精确多边形（False）
- `confidence_threshold` (FLOAT)：置信度阈值（0.0-1.0，默认 0.5）
- `exclude_words` (STRING)：排除词列表（每行一个）

**输出**：
- `mask` (MASK)：文字区域遮罩
- `detected_text` (STRING)：所有识别到的文字
- `debug_image` (IMAGE)：带边界框标注的调试图像
- `bounding_boxes` (LIST)：边界框详细信息列表
