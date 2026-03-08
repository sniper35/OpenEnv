"""FastAPI app for the IsaacLab Arena OpenEnv wrapper."""

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import IsaacLabArenaAction, IsaacLabArenaObservation
    from .isaaclab_arena_environment import IsaacLabArenaEnvironment
except ImportError:
    try:
        from isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
        )
        from isaaclab_arena_env.server.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from openenv.core.env_server.http_server import create_app
    except ImportError:
        from envs.isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
        )
        from envs.isaaclab_arena_env.server.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from openenv.core.env_server.http_server import create_app

app = create_app(
    IsaacLabArenaEnvironment,
    IsaacLabArenaAction,
    IsaacLabArenaObservation,
    env_name="isaaclab_arena_env",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()
