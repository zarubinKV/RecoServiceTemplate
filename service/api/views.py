from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request
from pydantic import BaseModel

from models.config import UserKnn_model
from models.load_models import POPULAR, USERKNN
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.auth.authorization import JWTBearer
from service.config import config
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class Message(BaseModel):
    message: str


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
    dependencies=[Depends(JWTBearer())],
)
async def health() -> str:
    return "36.6"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    dependencies=[Depends(JWTBearer())],
    responses={
        200: {"response": RecoResponse},
        403: {"response": Message},
        404: {"response": Message},
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    if model_name not in config.models:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if model_name == 'test_model':
        k_recs = request.app.state.k_recs
        reco = list(range(k_recs))
    elif model_name == 'userknn_model':
        # Online
        if UserKnn_model.online:
            model = USERKNN['model']
            reco = model.predict_online(user_id, UserKnn_model.N_recs)
        # Offline
        else:
            userknn_reco_df = USERKNN['reco_df']
            reco = list(
                userknn_reco_df[userknn_reco_df.user_id == user_id].item_id,
            )
        popular_reco_df = POPULAR['reco_df']
        reco_popular = list(popular_reco_df.item_id)
        i = 0
        # Model for cold users: Popular()
        # Model blending (userkNN.item_id union Popular.item_id limit N_recs)
        while len(reco) < UserKnn_model.N_recs:
            if reco_popular[i] not in reco:
                reco.append(reco_popular[i])
            i += 1
    elif model_name == 'popular_model':
        popular_reco_df = POPULAR['reco_df']
        reco = list(popular_reco_df.item_id)

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
