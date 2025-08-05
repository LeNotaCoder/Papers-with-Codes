import av
import torch
import torchvision.transforms as T

def read_video_frames(video_path, resize=(216, 216)):
    frames = []
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Error: Could not open video file {video_path} - {e}")
        return None

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(resize),
        T.ToTensor()
    ])

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        img = img[:, :, ::-1].copy() 
        img = torch.from_numpy(img).permute(2, 0, 1) 
        img = transform(img) 
        frames.append(img)

    if not frames:
        print(f"No frames extracted from {video_path}")
        return None

    return torch.stack(frames)  


def show_tensor_as_image(tensor):
    img = tensor[0].detach().cpu()
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min + 1e-8)
    img_np = img.permute(1,2,0).numpy()
    img_uint8 = (img_np * 255).astype("uint8")
    
    return img_uint8