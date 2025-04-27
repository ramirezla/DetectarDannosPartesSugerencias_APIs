# api_dannos/main.py
from api.app import app
import os
from fastapi import FastAPI

app = FastAPI()
port = int(os.environ.get("PORT", 10000))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)