"""Script to run the FastAPI server for Wise Nutrition."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("wise_nutrition.api:app", host="0.0.0.0", port=8000, reload=True)
