import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as ApiPath
from pydantic import BaseModel, ConfigDict, Field, model_validator

from movie_recs.service import Recommender


class RatingValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rating: float = Field(ge=0.5, le=5.0, multiple_of=0.5)


class MovieRating(RatingValue):
    movie_id: int = Field(gt=0)


class CreateUser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ratings: list[MovieRating] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def unique_movies(self):
        ids = [rating.movie_id for rating in self.ratings]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate movie_id in ratings")
        return self


def create_app(artifact_dir: Path | None = None, user_db: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.recommender = Recommender(
            artifact_dir or Path(os.getenv("ARTIFACT_DIR", "artifacts")),
            user_db or Path(os.getenv("USER_DB", "state/users.sqlite3")),
        )
        yield

    app = FastAPI(title="Movie recommendations", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        service = request.app.state.recommender
        return {"status": "ok", "default_model": service.default_model}

    @app.get("/models")
    def models(request: Request) -> dict:
        service = request.app.state.recommender
        return {
            "default_model": service.default_model,
            "models": {
                name: {"validation": result["validation"], "test": result["test"]}
                for name, result in service.report["models"].items()
            },
        }

    @app.get("/movies")
    def movies(
        request: Request,
        query: Annotated[str, Query(max_length=200)] = "",
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> dict:
        """Find films in the trained catalog before submitting ratings."""
        return {"movies": request.app.state.recommender.movies(query, limit)}

    @app.post("/users", status_code=201)
    def create_user(request: Request, body: CreateUser) -> dict:
        """Create a persistent profile, optionally with initial ratings."""
        service = request.app.state.recommender
        try:
            user_id = service.create_user({row.movie_id: row.rating for row in body.ratings})
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"user_id": user_id, "ratings": [row.model_dump() for row in body.ratings]}

    @app.put("/users/{user_id}/ratings/{movie_id}")
    def rate(
        request: Request,
        body: RatingValue,
        user_id: Annotated[int, ApiPath(gt=0)],
        movie_id: Annotated[int, ApiPath(gt=0)],
    ) -> dict:
        """Add or replace one rating; recommendations update on the next request."""
        try:
            request.app.state.recommender.rate(user_id, movie_id, body.rating)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown user_id") from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"user_id": user_id, "movie_id": movie_id, "rating": body.rating}

    @app.get("/recommendations/{user_id}")
    def recommendations(
        request: Request,
        user_id: Annotated[int, ApiPath(gt=0)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        model: Literal["mf", "neural", "als", "sansa"] | None = None,
    ) -> dict:
        try:
            return request.app.state.recommender.recommend(user_id, k, model)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown or untrained user_id") from exc

    return app


app = create_app()
