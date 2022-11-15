from typing import List

from fastapi import APIRouter, FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]

TOKEN_ACCESS = "12kuhGUh7yG76g"
class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer,
                                                                self).__call__(
            request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403,
                                    detail="Invalid authentication scheme.")
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403,
                                    detail="Invalid token or expired token.")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403,
                                detail="Invalid authorization code.")

    def verify_jwt(self, token: str) -> bool:
        isTokenValid: bool = False

        if token == TOKEN_ACCESS:
            isTokenValid = True
        return isTokenValid


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
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    # Write your code here

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs
    reco = list(range(k_recs))
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
