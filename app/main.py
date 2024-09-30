from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.routers import train, visualize

app = FastAPI(title = 'Anomaly Detection API')

#include the routers for different functionality
app.include_router(train.router, prefix = "/train", tags = ["Train"])
app.include_router(visualize.router, prefix = "/visualize", tags = ["Visualize"])

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code = 500,
        content = {"message": "An error occurred", "details": str(exc)},
    )

@app.get("/")
def read_root():
    return {"message" : "Welcome to Fraud Detection API!"}