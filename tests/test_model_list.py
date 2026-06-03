from pathlib import Path

from app.main import _discover_model_files


def test_discover_model_files_lists_pt_and_marks_active(tmp_path: Path) -> None:
    active_model = tmp_path / "mobile-fall.pt"
    other_model = tmp_path / "lightweight_2dcnn_best.pth"
    ignored_file = tmp_path / "notes.txt"
    active_model.write_bytes(b"active")
    other_model.write_bytes(b"other")
    ignored_file.write_text("ignore", encoding="utf-8")

    models = _discover_model_files(
        sources=[("MODEL_SEARCH_PATHS", tmp_path)],
        extensions={".pt", ".pth"},
        active_model_path=str(active_model.resolve()),
    )

    assert [model["file_name"] for model in models] == [
        "lightweight_2dcnn_best.pth",
        "mobile-fall.pt",
    ]
    assert any(model["active"] for model in models)
    assert all(model["file_name"] != "notes.txt" for model in models)
