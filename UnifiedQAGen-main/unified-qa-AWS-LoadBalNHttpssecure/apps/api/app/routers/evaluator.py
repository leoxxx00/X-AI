from fastapi import APIRouter, HTTPException
from openai import AuthenticationError
import requests

from app.schemas.evaluator import EvaluatorRequest, EvaluatorResponse
from app.services.evaluator import run_evaluator_service

router = APIRouter(prefix="/api/v1/evaluator", tags=["Evaluator"])


@router.post("", response_model=EvaluatorResponse)
def evaluate_url(request: EvaluatorRequest):
    try:
        result = run_evaluator_service(
            url_input=str(request.url),
            strictness=request.strictness,
        )
        return result

    except AuthenticationError:
        raise HTTPException(
            status_code=500,
            detail="DeepSeek authentication failed. Please verify DEEPSEEK_API_KEY.",
        )
    except requests.exceptions.RequestException as e:
        response = getattr(e, "response", None)
        status_code = getattr(response, "status_code", None)

        if status_code in {401, 403, 406, 429}:
            raise HTTPException(status_code=400, detail="This URL blocks automated access.")

        raise HTTPException(status_code=400, detail=f"Network or URL fetch error: {str(e)}")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")