"""Fine-tune YOLO on a produce dataset.

    python training/train_yolo.py --data training/data.yaml --epochs 80

Starting from COCO-pretrained weights (transfer learning) works well even with a
few hundred labelled images per class. After training, point the app at the
weights:  NUTRIVISION_YOLO_WEIGHTS=runs/detect/nutrivision/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained starting weights")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, name="nutrivision", patience=20,
        # Colour augmentation is kept mild: hue carries ripeness information.
        hsv_h=0.005, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, flipud=0.2, mosaic=1.0, close_mosaic=10,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
