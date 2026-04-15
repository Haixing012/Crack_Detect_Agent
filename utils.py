class PixelToRealConverter:
    def __init__(self):
        pass

    @staticmethod
    def calculate_gsd_by_drone(altitude_m: float, focal_length_mm: float, sensor_width_mm: float, image_width_px: int) -> float:
        """
        模式一：通过无人机/相机的物理参数计算 GSD (厘米/像素)
        公式: GSD = (飞行高度 * 传感器宽度) / (焦距 * 图像宽度)
        """
        # 将高度从米转换为毫米进行统一下计算，最后换算为厘米
        gsd_mm = (altitude_m * 1000 * sensor_width_mm) / (focal_length_mm * image_width_px)
        return gsd_mm / 10  # 返回 厘米/像素

    @staticmethod
    def calculate_gsd_by_reference(reference_real_cm: float, reference_pixel: float) -> float:
        """
        模式二：通过画面中的参照物估算 (比如标准的15cm宽车道线)
        """
        return reference_real_cm / reference_pixel

    @staticmethod
    def convert(pixel_value: float, gsd_cm_per_px: float) -> float:
        """
        将像素值转换为真实厘米
        """
        return round(pixel_value * gsd_cm_per_px, 2)

# ================= 测试与使用示例 =================
if __name__ == "__main__":
    converter = PixelToRealConverter()

    # 假设你的无人机在 5米 高度拍摄，镜头焦距 4.5mm，传感器宽度 6.4mm，图片宽度 640像素
    gsd = converter.calculate_gsd_by_drone(
        altitude_m=5.0, 
        focal_length_mm=4.5, 
        sensor_width_mm=6.4, 
        image_width_px=640
    )
    print(f"当前相机的GSD估算为: {gsd:.2f} 厘米/像素")

    # 假设 YOLO 识别到裂缝宽度为 9.18 像素 (也就是你上一轮日志里的数据)
    crack_width_px = 9.18
    real_width_cm = converter.convert(crack_width_px, gsd)
    
    print(f"YOLO识别宽度: {crack_width_px} 像素 -> 真实估算宽度: {real_width_cm} cm")