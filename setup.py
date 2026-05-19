from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.exists():
        return []

    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def collect_data_files() -> list[tuple[str, list[str]]]:
    data_files = [("data", ["data/food_db.json"])]

    for directory in ("templates", "static"):
        root = ROOT / directory
        if root.exists():
            files = [
                str(path.relative_to(ROOT))
                for path in root.rglob("*")
                if path.is_file()
            ]
            if files:
                data_files.append((directory, files))

    return data_files


setup(
    name="athleteedge-ai",
    version="0.1.0",
    description="Sport-specific Indian nutrition planning engine for athletes.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8")
    if (ROOT / "README.md").exists()
    else "",
    long_description_content_type="text/markdown",
    py_modules=[
        "app",
        "fooddb",
        "offline",
        "smart_planner",
        "train_model",
    ],
    include_package_data=True,
    data_files=collect_data_files(),
    install_requires=read_requirements(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "athleteedge-app=app:app.run",
            "athleteedge-train=train_model:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Flask",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
