""" Entrypoint of the application. """
import uvicorn


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run("src.inference.api.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
