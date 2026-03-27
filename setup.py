from setuptools import setup, find_packages

setup(
    name="robocasa-tasks",
    version="0.1.0",
    package_dir={"robocasa_tasks": "."},
    packages=["robocasa_tasks"] + [
        f"robocasa_tasks.{p}"
        for p in find_packages(where=".")
    ],
    install_requires=[
        "mani-skill>=3.0.0b14",
        "numpy",
    ],
)
