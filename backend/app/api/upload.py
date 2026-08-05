from fastapi import APIRouter

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)
@router.get("/")
def upload_home():
    return {
        "message": "Upload API Working"
    }