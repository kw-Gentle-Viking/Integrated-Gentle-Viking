from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.db import get_db
from app.dependencies import get_current_user
from app.models import User
from app.schemas import UserCreate, UserRead, UserProfileUpdate
from app.crud_users import create_user, get_user, list_users, update_user_profile
from app.services_investment import calculate_risk_score
from app.security_guards import require_debug_endpoints

router = APIRouter()


@router.post("", response_model=UserRead, status_code=201)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        risk_score = calculate_risk_score(
            investment_goal=payload.investment_goal,
            investment_period=payload.investment_period,
            risk_tolerance=payload.risk_tolerance,
            investment_experience=payload.investment_experience,
            volatility_preference=payload.volatility_preference,
        )
        return create_user(db, payload, risk_score)
    except IntegrityError as e:
        db.rollback()
        err = str(e.orig).lower()
        if "email" in err:
            raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다")
        elif "username" in err:
            raise HTTPException(status_code=409, detail="이미 사용 중인 닉네임입니다")
        raise HTTPException(status_code=409, detail="중복된 값이 있습니다")


@router.get("/me", response_model=UserRead)
def read_my_profile(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserRead)
def update_profile(
    payload: UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    investment_field_names = [
        "investment_goal",
        "investment_period",
        "risk_tolerance",
        "investment_experience",
        "volatility_preference",
    ]
    risk_score = None

    if any(name in payload.model_fields_set for name in investment_field_names):
        values = {
            name: getattr(payload, name) if getattr(payload, name) is not None else getattr(current_user, name)
            for name in investment_field_names
        }
        if any(value is None for value in values.values()):
            raise HTTPException(status_code=422, detail="투자성향 항목을 모두 입력해주세요")
        risk_score = calculate_risk_score(**values)

    return update_user_profile(db, current_user, payload, risk_score)


# 본인 정보만 조회 가능 (예전에는 로그인 없이 누구나 임의 회원의 이메일/전화/생년월일을 볼 수 있었다)
@router.get("/{user_id}", response_model=UserRead)
def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# 개발/디버깅용: ENABLE_DEBUG_ENDPOINTS=true 이고 로그인한 경우에만 (기본은 404)
@router.get("", response_model=list[UserRead], dependencies=[Depends(require_debug_endpoints)])
def read_users(
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_users(db, limit=limit)
