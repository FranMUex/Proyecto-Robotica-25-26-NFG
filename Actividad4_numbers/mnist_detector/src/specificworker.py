#!/usr/bin/python3

import itertools
import math
import os
import sys
import time
import traceback

import cv2
import interfaces as ifaces
import MODMNIST
import numpy as np
import torch
import torch.nn as nn

from genericworker import *
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from rich.console import Console

sys.path.append("/opt/robocomp/lib")
console = Console(highlight=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, configData, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map, configData)
        self.Period = configData["Period"]["Compute"]

        self.model = SimpleCNN()
        self.model.eval()

        model_path = os.path.join(os.path.dirname(__file__), "mnist_model.pth")
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=torch.device("cpu"))
                )
                console.print(f"[green]Model loaded from {model_path}[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load model weights: {e}[/yellow]")
        else:
            console.print("[yellow]No pretrained model found. Using random weights.[/yellow]")

        if startup_check:
            self.startup_check()
            return

        started_camera = False
        while not started_camera:
            try:
                self.rgb_original = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
                started_camera = True
                print("Connected to Camera360RGB")
            except Ice.Exception:
                print("Retrying Camera360RGB connection...")
                time.sleep(1)

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)


    @QtCore.Slot()
    def compute(self):
        try:
            image = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
            color = np.frombuffer(image.image, dtype=np.uint8).reshape(
                image.height, image.width, 3
            )

            rect = self.detect_frame(color)
            color_copy = color.copy()

            if rect is not None:
                x1, y1, x2, y2 = rect
                cv2.rectangle(color_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

                result = self.MNIST_getNumber()
                if result.label != -1:
                    cv2.putText(
                        color_copy,
                        f"Digit: {result.label} ({result.confidence:.2f})",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Camera360RGB", color_copy)
            cv2.waitKey(1)

        except Exception as e:
            console.print(f"[red]Error in compute: {e}[/red]")
            traceback.print_exc()


    def detect_frame(self, color):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        h, w = gray.shape
        best = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.002 * w * h:
                continue

            x, y, bwc, bhc = cv2.boundingRect(cnt)
            aspect = bwc / float(bhc)

            if 0.4 <= aspect <= 2.5:
                roi = bw[y : y + bhc, x : x + bwc]
                white_ratio = cv2.countNonZero(roi) / float(bwc * bhc)
                score = white_ratio * area

                if score > best_score:
                    best_score = score
                    best = (x, y, bwc, bhc)

        if best is None:
            return None

        x, y, bwc, bhc = best
        margin = int(min(bwc, bhc) * 0.05)

        return [
            max(0, x + margin) +10,
            max(0, y + margin) +10,
            min(w, x + bwc - margin),
            min(h, y + bhc - margin),
        ]


    def MNIST_getNumber(self):
        try:
            image = self.camera360rgb_proxy.getROI(-1, -1, -1, -1, -1, -1)
            color = np.frombuffer(image.image, dtype=np.uint8).reshape(
                image.height, image.width, 3
            )

            rect = self.detect_frame(color)
            if rect is None:
                return MODMNIST.MNISTResult(label=-1, confidence=-1.0)

            x1, y1, x2, y2 = rect
            roi = color[y1:y2, x1:x2]

            gray = roi.mean(axis=2).astype(np.uint8)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            _, bw = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            contours, _ = cv2.findContours(
                bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return MODMNIST.MNISTResult(label=-1, confidence=-1.0)

            h_img, w_img = bw.shape
            digit_cnt = None
            best_area = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)

                x, y, w, h = cv2.boundingRect(cnt)

                if x <= 2 or y <= 2 or x + w >= w_img - 2 or y + h >= h_img - 2:
                    continue

                if area > best_area:
                    best_area = area
                    digit_cnt = cnt

            if digit_cnt is None:
                return MODMNIST.MNISTResult(label=-1, confidence=-1.0)

            x, y, w, h = cv2.boundingRect(digit_cnt)
            digit = bw[y:y+h, x:x+w]

            side = max(28, int(max(w, h) * 1.5))
            canvas = np.zeros((side, side), dtype=np.uint8)

            x_off = (side - w) // 2
            y_off = (side - h) // 2
            canvas[y_off:y_off + h, x_off:x_off + w] = digit

            img = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

            img = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                out = self.model(tensor)
                probs = torch.softmax(out, dim=1)
                pred = out.argmax(1).item()
                conf = probs[0, pred].item()

            console.print(f"[cyan]Pred: {pred}  Conf: {conf:.2f}[/cyan]")

            if conf < 0.6:
                return MODMNIST.MNISTResult(label=-1, confidence=float(conf))

            return MODMNIST.MNISTResult(label=pred, confidence=float(conf))

        except Exception as e:
            console.print(f"[red]MNIST_getNumber error: {e}[/red]")
            traceback.print_exc()
            return MODMNIST.MNISTResult(label=-1, confidence=-1.0)


    def startup_check(self):
        print("Startup check OK")
        QTimer.singleShot(200, QApplication.instance().quit)
