"""PySide6 desktop app (Phase 3) wrapping the pipeline.

Run:  python -m lecturenotes.app

Drag a lecture video in (or Browse), optionally add the lecturer's slides/notes,
set the API key once, hit Start. The pipeline runs on a background thread so the
window stays responsive, reporting per-stage progress.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from .config import Config, resolve_api_key, save_api_key
from .pipeline import run_pipeline

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
MATERIAL_EXTS = {".pdf", ".pptx", ".ppt"}

# relative weight of each stage for the overall progress bar (sums to ~1.0)
STAGE_WEIGHT = {
    "materials": 0.03, "ingest": 0.04, "transcribe": 0.40, "frames": 0.08,
    "vision": 0.20, "align": 0.02, "synthesize": 0.20, "admin": 0.02, "pptx": 0.01,
}
STAGE_LABEL = {
    "materials": "Loading reference materials…",
    "ingest": "Extracting audio…",
    "transcribe": "Transcribing on GPU…",
    "frames": "Scanning for slides/board changes…",
    "vision": "Reading on-screen content…",
    "align": "Aligning transcript…",
    "synthesize": "Writing notes…",
    "admin": "Collecting announcements…",
    "pptx": "Building PowerPoint…",
    "done": "Done",
}
_TOTAL_WEIGHT = sum(STAGE_WEIGHT.values())


class PipelineWorker(QThread):
    progress = Signal(str, float)
    done = Signal(str, int, int)
    failed = Signal(str)

    def __init__(self, video, out, materials, cfg):
        super().__init__()
        self._video, self._out, self._materials, self._cfg = video, out, materials, cfg

    def run(self):
        try:
            res = run_pipeline(
                self._video, self._out, self._cfg, materials_path=self._materials,
                progress=lambda s, f: self.progress.emit(s, f))
            self.done.emit(res.output_path or str(self._out), len(res.slides), len(res.admin))
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr)  # full detail in the terminal
            log_note = ""
            try:
                logp = Path(self._out).with_name("lecturenotes_error.log")
                logp.write_text(tb, encoding="utf-8")
                log_note = f"\n\nFull details saved to:\n{logp}"
            except Exception:
                pass
            self.failed.emit(f"{e}{log_note}\n\n{tb}")


class DropFrame(QFrame):
    dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(110)
        self.setStyleSheet(
            "DropFrame{border:2px dashed #c3b9f0;border-radius:12px;background:#faf9ff;}")
        lay = QVBoxLayout(self)
        self.label = QLabel("Drag a lecture video here, or Browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color:#6b7280;border:none;background:transparent;")
        lay.addWidget(self.label)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p:
                self.dropped.emit(p)
                break


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lecture Notes")
        self.resize(640, 560)
        self.video_path: Path | None = None
        self.materials_path: Path | None = None
        self.out_path: Path | None = None
        self._last_output: Path | None = None
        self._frac: dict[str, float] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        title = QLabel("Lecture → Notes")
        f = QFont(); f.setPointSize(18); f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        self.drop = DropFrame()
        self.drop.dropped.connect(self._on_drop)
        root.addWidget(self.drop)

        root.addLayout(self._file_row("Video:", self._pick_video, "video_lbl"))
        root.addLayout(self._file_row("Materials (optional):", self._pick_materials,
                                      "mat_lbl", clear=self._clear_materials))
        root.addLayout(self._file_row("Output folder:", self._pick_outdir, "out_lbl"))

        # API key row
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API key:"))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-ant-…  (stored locally)")
        existing = resolve_api_key()
        if existing:
            self.key_edit.setText(existing)
        self.key_edit.textChanged.connect(lambda _: self._refresh_start())
        key_row.addWidget(self.key_edit, 1)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_key)
        key_row.addWidget(save_btn)
        root.addLayout(key_row)

        self.quality = QCheckBox("Higher quality (use the stronger model on hard frames)")
        self.quality.setChecked(True)
        root.addWidget(self.quality)

        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self._start)
        root.addWidget(self.start_btn)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        root.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#6b7280;")
        root.addWidget(self.status)

        self.open_row = QHBoxLayout()
        self.open_file_btn = QPushButton("Open deck")
        self.open_file_btn.clicked.connect(self._open_file)
        self.open_dir_btn = QPushButton("Open folder")
        self.open_dir_btn.clicked.connect(self._open_dir)
        for b in (self.open_file_btn, self.open_dir_btn):
            b.setEnabled(False)
            self.open_row.addWidget(b)
        root.addLayout(self.open_row)
        root.addStretch(1)

        self._refresh_start()

    # ---- UI helpers ----
    def _file_row(self, label, handler, attr, clear=None):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        lbl = QLabel("—")
        lbl.setStyleSheet("color:#374151;")
        setattr(self, attr, lbl)
        row.addWidget(lbl, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(handler)
        row.addWidget(btn)
        if clear:
            cbtn = QPushButton("Clear")
            cbtn.clicked.connect(clear)
            row.addWidget(cbtn)
        return row

    def _on_drop(self, path: str):
        ext = Path(path).suffix.lower()
        if ext in VIDEO_EXTS:
            self._set_video(Path(path))
        elif ext in MATERIAL_EXTS:
            self._set_materials(Path(path))
        else:
            QMessageBox.warning(self, "Unsupported", f"Not a video or materials file:\n{path}")

    def _set_video(self, p: Path):
        self.video_path = p
        self.video_lbl.setText(p.name)
        self.drop.label.setText(p.name)
        if not self.out_path:
            self.out_path = p.parent
            self.out_lbl.setText(str(p.parent))
        self._refresh_start()

    def _set_materials(self, p: Path):
        self.materials_path = p
        self.mat_lbl.setText(p.name)

    def _clear_materials(self):
        self.materials_path = None
        self.mat_lbl.setText("—")

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Choose lecture video", "",
            "Video (*.mp4 *.mkv *.mov *.avi *.webm *.m4v)")
        if p:
            self._set_video(Path(p))

    def _pick_materials(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Choose slides/notes", "", "Slides or PDF (*.pdf *.pptx *.ppt)")
        if p:
            self._set_materials(Path(p))

    def _pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.out_path = Path(d)
            self.out_lbl.setText(d)

    def _save_key(self):
        key = self.key_edit.text().strip()
        if not key:
            return
        ok = save_api_key(key)
        QMessageBox.information(
            self, "API key",
            "Saved to your system keyring." if ok else
            "Couldn't reach the keyring; the key will still be used for this session.")

    def _refresh_start(self):
        ready = bool(self.video_path) and bool(self.key_edit.text().strip())
        self.start_btn.setEnabled(ready)

    # ---- run ----
    def _start(self):
        if not self.video_path:
            return
        out_dir = self.out_path or self.video_path.parent
        out_pptx = out_dir / (self.video_path.stem + ".pptx")

        cfg = Config(api_key=self.key_edit.text().strip(),
                     allow_escalation=self.quality.isChecked())

        self._frac.clear()
        self.bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.open_file_btn.setEnabled(False)
        self.open_dir_btn.setEnabled(False)
        self.status.setText("Starting…")

        self.worker = PipelineWorker(
            self.video_path, out_pptx,
            str(self.materials_path) if self.materials_path else None, cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, stage: str, frac: float):
        if stage == "done":
            self.bar.setValue(100)
            return
        self._frac[stage] = frac
        overall = sum(STAGE_WEIGHT.get(s, 0) * f for s, f in self._frac.items())
        self.bar.setValue(int(100 * overall / _TOTAL_WEIGHT))
        self.status.setText(STAGE_LABEL.get(stage, stage))

    def _on_done(self, out_path: str, slides: int, admin: int):
        self._last_output = Path(out_path)
        self.bar.setValue(100)
        self.status.setText(f"Done — {slides} slides + {admin} announcements.")
        self.start_btn.setEnabled(True)
        self.open_file_btn.setEnabled(True)
        self.open_dir_btn.setEnabled(True)

    def _on_failed(self, message: str):
        self.start_btn.setEnabled(True)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Pipeline error", message[:2000])

    def _open_file(self):
        if self._last_output and self._last_output.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))

    def _open_dir(self):
        if self._last_output:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output.parent)))


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
