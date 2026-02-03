# main.py
import os
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from train_model import ForkedResNet50
from torchvision.transforms import transforms
from PIL import Image, ImageOps, ImageEnhance
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import cv2
import numpy as np
import hashlib
import logging
from PIL.ExifTags import TAGS
import base64

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 配置部分 ---
MODEL_PATH = "best_model.pth"
MAP_PATH = "char_map.json"

# --- 全局初始化模型（启动时加载一次） ---
class Recognizer:
    def __init__(self, model_path: str = MODEL_PATH, map_path: str = MAP_PATH):
        if not os.path.exists(model_path) or not os.path.exists(map_path):
            raise FileNotFoundError(
                f"模型文件 '{model_path}' 或字符映射表 '{map_path}' 不存在，"
                "请先运行 train_model.py 进行模型训练。"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载字符映射
        with open(map_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        num_classes = len(self.char_to_idx)

        # 构建并加载模型
        self.model = self._get_model(num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        # Support checkpoints in both formats: {'model_state_dict': ...} or raw state_dict
        # Works with both old training runs and new improved training
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (disables dropout)

        # 定义图像预处理 - match training validation pipeline (Resize 256 + CenterCrop 224)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_model(self, num_classes: int) -> nn.Module:
        """
        Initialize the model architecture (ForkedResNet50 with pretrained backbone).
        
        Model features:
        - Pretrained ResNet50 backbone (ImageNet weights) for better feature transfer
        - Shared feature extraction layers + task-specific layer4 copies
        - MLP heads with dropout (0.5) for character and style classification
        - Better generalization due to regularization
        
        To resume from a checkpoint:
          checkpoint_path = 'latest_checkpoint.pth'
          checkpoint = torch.load(checkpoint_path)
          if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
              state_dict = checkpoint['model_state_dict']
          else:
              state_dict = checkpoint
          model.load_state_dict(state_dict)
        """
        model = ForkedResNet50(num_classes)
        return model

    def preprocess_image(self, image_bytes: bytes, user_agent: str = "") -> Image.Image:
        """统一的图像预处理，特别处理移动设备上传的图片，转换为灰度以提高模型性能"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # 记录原始图像信息用于调试
            logger.info(f"原始图像 - 格式: {image.format}, 模式: {image.mode}, 尺寸: {image.size}")
            
            # 处理EXIF方向信息（手机照片常有旋转问题）
            try:
                exif = image._getexif()
                if exif:
                    for tag, value in exif.items():
                        decoded = TAGS.get(tag, tag)
                        if decoded == 'Orientation':
                            if value == 3:
                                image = image.rotate(180, expand=True)
                            elif value == 6:
                                image = image.rotate(270, expand=True)
                            elif value == 8:
                                image = image.rotate(90, expand=True)
                            break
            except Exception as e:
                logger.warning(f"EXIF处理失败: {e}")
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 检测是否为移动设备并应用增强处理
            is_mobile = any(mobile_indicator in user_agent.lower() 
                           for mobile_indicator in ['mobile', 'iphone', 'android', 'ipad'])
            
            if is_mobile:
                logger.info("检测到移动设备，应用增强预处理")
                # 增强对比度
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
                # 轻微锐化
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.1)
            
            # 转换为灰度图以减少颜色干扰，提高模型泛化能力
            logger.info("转换为灰度图")
            image = image.convert('L')  # Convert to grayscale (8-bit single channel)
            # 将灰度图转换回RGB格式（3通道）以保持与模型输入兼容
            image = image.convert('RGB')  # Repeat grayscale channel 3 times
            
            return image
            
        except Exception as e:
            raise ValueError(f"无法处理图片: {e}")

    def assess_image_quality(self, image: Image.Image) -> Dict:
        """评估图像质量"""
        # 转换为numpy数组进行处理
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            
        # 计算清晰度（拉普拉斯方差）
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算亮度和对比度
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        quality_info = {
            'clarity': float(clarity),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_acceptable': clarity > 50 and 30 < brightness < 220
        }
        
        logger.info(f"图像质量评估: {quality_info}")
        return quality_info

    def recognize(self, image_bytes: bytes, top_k: int = 5, user_agent: str = "") -> List[Dict[str, str]]:
        """
        识别上传的图像。

        Args:
            image_bytes: 图片的二进制数据。
            top_k: 返回前k个结果。
            user_agent: 用户代理字符串，用于设备检测

        Returns:
            一个字典列表，每个字典包含 `char` 和 `prob` 键。
        """
        # 记录图像哈希用于调试
        image_hash = hashlib.md5(image_bytes).hexdigest()
        logger.info(f"处理图像 - 哈希: {image_hash}, 设备: {user_agent}")

        # 预处理图像
        image = self.preprocess_image(image_bytes, user_agent)
        
        # 评估图像质量
        quality_info = self.assess_image_quality(image)
        if not quality_info['is_acceptable']:
            logger.warning(f"图像质量可能影响识别: {quality_info}")

        # 应用模型预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            # ForkedResNet50 returns (char_logits, style_logits)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                char_logits, style_logits = outputs
            else:
                char_logits = outputs

            probabilities = torch.nn.functional.softmax(char_logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        top_probs_np = top_probs.cpu().detach().numpy().flatten()
        top_indices_np = top_indices.cpu().detach().numpy().flatten()

        for i in range(top_k):
            char_idx = top_indices_np[i]
            char_name = self.idx_to_char.get(char_idx, '?')
            probability = top_probs_np[i]
            results.append({
                'char': char_name,
                'prob': f'{probability:.2%}'
            })

        logger.info(f"识别结果: {results}")
        return results

    def visualize(self, image_bytes: bytes, user_agent: str = "", target='char') -> Dict:
        """Return processed image, saliency gradient and predictions.

        Returns a dict with keys: 'processed_base64', 'saliency_base64', 'predictions'.
        """
        if target not in ('char', 'style'):
            target = 'char'

        image = self.preprocess_image(image_bytes, user_agent)
        orig_w, orig_h = image.size

        # Prepare tensor with gradient enabled
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        img_t.requires_grad_(True)

        # Forward pass
        char_logits, style_logits = self.model(img_t)

        # Choose target prediction index
        probs_char = torch.softmax(char_logits, dim=1)
        probs_style = torch.softmax(style_logits, dim=1)
        char_idx = int(torch.argmax(probs_char, dim=1)[0].item())
        style_idx = int(torch.argmax(probs_style, dim=1)[0].item())

        # Compute gradients w.r.t. input for selected prediction
        self.model.zero_grad()
        if target == 'char':
            score = char_logits[0, char_idx]
        else:
            score = style_logits[0, style_idx]
        score.backward()

        # Extract input gradient (saliency map)
        input_grad = img_t.grad.detach().cpu().numpy()  # shape: (1, 3, 224, 224)

        # Compute saliency as max absolute gradient across channels
        saliency = np.max(np.abs(input_grad[0]), axis=0)  # shape: (224, 224)

        # Normalize saliency to [0, 1]
        saliency = np.maximum(saliency, 0)
        if saliency.max() > 0:
            saliency_norm = saliency / saliency.max()
        else:
            saliency_norm = saliency

        # Convert to 8-bit grayscale
        saliency_gray = (saliency_norm * 255).astype(np.uint8)
        saliency_gray = cv2.resize(saliency_gray, (orig_w, orig_h))

        # Apply JET colormap for visualization
        saliency_colored = cv2.applyColorMap(saliency_gray, cv2.COLORMAP_JET)
        saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)

        # Encode processed image as PNG base64
        pil_proc = image.convert('RGB')
        proc_buf = io.BytesIO()
        pil_proc.save(proc_buf, format='PNG')
        proc_b64 = base64.b64encode(proc_buf.getvalue()).decode('ascii')

        # Encode saliency as PNG base64
        saliency_pil = Image.fromarray(saliency_colored)
        sal_buf = io.BytesIO()
        saliency_pil.save(sal_buf, format='PNG')
        sal_b64 = base64.b64encode(sal_buf.getvalue()).decode('ascii')

        # Font style mapping
        style_mapping = {0: 'cs', 1: 'ks', 2: 'ls', 3: 'xs', 4: 'zs'}
        style_name_cn = {
            'cs': '草书',
            'ks': '楷书',
            'ls': '隶书',
            'xs': '行书',
            'zs': '篆书'
        }

        # Prepare predictions
        results = []
        topk = 5
        top_probs, top_indices = torch.topk(probs_char, topk)
        for p, idx in zip(top_probs[0].detach().cpu().numpy(), top_indices[0].detach().cpu().numpy()):
            results.append({'char': self.idx_to_char.get(int(idx), '?'), 'prob': float(p)})

        style_results = []
        topk2 = min(5, probs_style.shape[1])
        top_probs_s, top_indices_s = torch.topk(probs_style, topk2)
        for p, idx in zip(top_probs_s[0].detach().cpu().numpy(), top_indices_s[0].detach().cpu().numpy()):
            style_abbr = style_mapping.get(int(idx), '?')
            style_cn = style_name_cn.get(style_abbr, '?')
            style_results.append({'style': style_cn, 'prob': float(p)})

        style_abbr_pred = style_mapping.get(style_idx, '?')
        style_cn_pred = style_name_cn.get(style_abbr_pred, '?')

        return {
            'processed_base64': proc_b64,
            'saliency_base64': sal_b64,
            'char_results': results,
            'style_results': style_results,
            'char_pred': self.idx_to_char.get(char_idx, '?'),
            'char_conf': float(probs_char[0, char_idx].detach().item()),
            'style_pred': style_cn_pred,
            'style_conf': float(probs_style[0, style_idx].detach().item())
        }

# --- FastAPI 应用初始化 ---
app = FastAPI(
    title="汉字书法字体识别",
    description="上传一张汉字图片，返回识别出的汉字及其置信度。",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化识别器（全局单例）
try:
    recognizer = Recognizer()
    logger.info("模型加载成功，服务已启动")
except FileNotFoundError as e:
    logger.error(f"启动失败: {e}")
    recognizer = None

# --- API 路由 ---
@app.post("/upload", response_model=List[Dict[str, str]])
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    上传图片进行汉字识别。

    - **file**: 需要识别的图片文件 (jpg, png, etc.)
    """
    if not recognizer:
        raise HTTPException(
            status_code=503,
            detail="服务暂不可用，模型文件未找到。"
        )

    # 检查文件类型
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上传的文件必须是图片格式。")

    try:
        image_bytes = await file.read()
        user_agent = request.headers.get('User-Agent', '')
        results = recognizer.recognize(image_bytes, top_k=5, user_agent=user_agent)
        return results
    except ValueError as ve:
        logger.error(f"图片处理错误: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"识别错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

# 调试接口
@app.post("/debug_upload")
async def debug_upload(request: Request, file: UploadFile = File(...)):
    """调试接口，返回上传图片的详细信息"""
    try:
        image_bytes = await file.read()
        user_agent = request.headers.get('User-Agent', '')
        
        # 使用recognizer的预处理方法来分析图片
        image = recognizer.preprocess_image(image_bytes, user_agent)
        quality_info = recognizer.assess_image_quality(image)
        
        debug_info = {
            'file_size': len(image_bytes),
            'file_hash': hashlib.md5(image_bytes).hexdigest(),
            'image_size': image.size,
            'image_mode': image.mode,
            'quality_assessment': quality_info,
            'user_agent': user_agent
        }
        
        return JSONResponse(content=debug_info)
    except Exception as e:
        logger.error(f"调试接口错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize")
async def visualize_upload(request: Request, file: UploadFile = File(...)):
    """Upload an image and return processed image, overlay heatmap, and predictions."""
    if not recognizer:
        raise HTTPException(status_code=503, detail="Service unavailable: model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Uploaded file must be an image')

    try:
        image_bytes = await file.read()
        user_agent = request.headers.get('User-Agent', '')
        result = recognizer.visualize(image_bytes, user_agent=user_agent, target='char')
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"visualize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="web")

@app.get("/")
async def root():
    return {"message": "欢迎使用汉字书法识别模型，请使用 POST /upload 接口上传图片。"}

@app.get("/health")
async def health_check():
    """健康检查接口"""
    status = "healthy" if recognizer else "unhealthy"
    return {
        "status": status,
        "model_loaded": recognizer is not None
    }

if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
